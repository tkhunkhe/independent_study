#Identify pupils. Based on beta 1

import numpy as np
import cv2
import time

cap = cv2.VideoCapture(0) 	#640,480 # from webcam
w = 640
h = 480

while(cap.isOpened()):
    ret, frame = cap.read() # read frame from webcam, return ret = true if frame is read correctly
    if ret==True:
	
		#downsample
		#frameD = cv2.pyrDown(cv2.pyrDown(frame))
		#frameDBW = cv2.cvtColor(frameD,cv2.COLOR_RGB2GRAY)
	
		#detect face
		frame = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
		faces = cv2.CascadeClassifier('haarcascade_eye.xml')
		detected = faces.detectMultiScale(frame, 1.3, 5) # Detects objects of different sizes in the input image. The detected objects are returned as a list of rectangles.
	
		#faces = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
		#detected2 = faces.detectMultiScale(frameDBW, 1.3, 5)
		
		pupilFrame = frame
		pupilO = frame
		windowClose = np.ones((5,5),np.uint8) # unit range = [0,255]
		windowOpen = np.ones((2,2),np.uint8)
		windowErode = np.ones((2,2),np.uint8)

		#draw square
		for (x,y,w,h) in detected: # for all the 'eye' detected in from faces.detectMultiScale
			cv2.rectangle(frame, (x,y), ((x+w),(y+h)), (0,0,255),1)	# cv.Rectangle(img, pt1, pt2, color, thickness=1, lineType=8, shift=0) # note that point is (x,y)
			cv2.line(frame, (x,y), ((x+w,y+h)), (0,0,255),1) # diagonal line 
			cv2.line(frame, (x+w,y), ((x,y+h)), (0,0,255),1) # anti-diagonal line
			pupilFrame = cv2.equalizeHist(frame[y+(h*.25):(y+h), x:(x+w)]) # Equalizes the histogram of a grayscale image. # The algorithm normalizes the brightness and increases the contrast of the image.

			pupilO = pupilFrame # equalized pupilFrame
			ret, pupilFrame = cv2.threshold(pupilFrame,55,255,cv2.THRESH_BINARY)		#50 ..nothin 70 is better # to get binary image out of a grayscale image
			
			# Performs advanced morphological transformations. CLOSE ERODE OPEN
			# read: http://docs.opencv.org/2.4/doc/tutorials/imgproc/opening_closing_hats/opening_closing_hats.html
			pupilFrame = cv2.morphologyEx(pupilFrame, cv2.MORPH_CLOSE, windowClose) 
			pupilFrame = cv2.morphologyEx(pupilFrame, cv2.MORPH_ERODE, windowErode)
			pupilFrame = cv2.morphologyEx(pupilFrame, cv2.MORPH_OPEN, windowOpen)

			#so above we do image processing to get the pupil..
			#now we find the biggest blob and get the centriod
			
			threshold = cv2.inRange(pupilFrame,250,255)		#get the blobs, # Checks if array elements lie between the elements of two other arrays.

			contours, hierarchy = cv2.findContours(threshold,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE) # Python: cv.FindContours(image, storage, mode=CV_RETR_LIST, method=CV_CHAIN_APPROX_SIMPLE, offset=(0, 0)) -> contours
			
			#if there are 3 or more blobs, delete the biggest and delete the left most for the right eye
			#if there are 2 blob, take the second largest
			#if there are 1 or less blobs, do nothing
			
			if len(contours) >= 2:
				#find biggest blob
				maxArea = 0
				MAindex = 0			#to get the unwanted frame 
				distanceX = []		#delete the left most (for right eye)
				currentIndex = 0 
				for cnt in contours:
					area = cv2.contourArea(cnt)
					center = cv2.moments(cnt)
					cx,cy = int(center['m10']/center['m00']), int(center['m01']/center['m00'])
					distanceX.append(cx)	
					if area > maxArea:
						maxArea = area
						MAindex = currentIndex
					currentIndex = currentIndex + 1
		
				del contours[MAindex]		#remove the picture frame contour
				del distanceX[MAindex]
			
			eye = 'right'

			if len(contours) >= 2:		#delete the left most blob for right eye
				if eye == 'right':
					edgeOfEye = distanceX.index(min(distanceX))
				else:
					edgeOfEye = distanceX.index(max(distanceX))	
				del contours[edgeOfEye]
				del distanceX[edgeOfEye]

			if len(contours) >= 1:		#get largest blob
				maxArea = 0
				for cnt in contours:
					area = cv2.contourArea(cnt)
					if area > maxArea:
						maxArea = area
						largeBlob = cnt
					
				
			if len(largeBlob) > 0:	
				center = cv2.moments(largeBlob)
				cx,cy = int(center['m10']/center['m00']), int(center['m01']/center['m00'])
				cv2.circle(pupilO,(cx,cy),5,255,-1) # fill white circle for pupil deteced area

	
		#show picture
		cv2.imshow('frame',pupilO) # eyes frame + pupil detected 
		cv2.imshow('frame2',pupilFrame) # contour 
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	
	#else:
		#break

# Release everything if job is finished
cap.release()
cv2.destroyAllWindows()





######## Sources of reading
## http://opencv-code.com/tutorials/pupil-detection-from-an-eye-image/
