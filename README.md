## Independent Study Project
- Spring '16
- Thanatcha Khunkhet (Kwan)
- Mentor/Advisor: Jesse Wang

Using pupillary dilatory patterns and micro-saccades to screen for neurological conditions before other symptoms manifest.

## The idea
Several neurological diseases are detrimental and most of them are difficult to detect at early stage. With the growth of amount of data recorded and data mining technology, screening features for neurological conditions before other symptoms manifest may be found. Thus, physicians may be able to cure the diseases at that early stage before it is too late. Previous studies have shown distinctive features of micro-saccades and pupil dilatory patterns in neurological impaired patients. In this study, we will try to see how far back we can go to observe the change in micro saccades and pupil dilatory patterns before a patient shows the classical symptoms of a neurological condition. We plan to mine on a collection of non-identified patient emr data, and use python’s OpenCV and SimpleCV package for image processing tasks. Data collection and pre-process may be our biggest problem. This depends on how clean the data are, and how much patients data we will have. Also, even though learning how to use OpenCV will take sometime, it is manageable.

## Problems encountered.
Although I was really interested in Machine Vision and this project, I had zero knowledge/experience of machine vision when we started. I've learned that there are many advanced math and theories behind image processing and object detection, and it's not just using a package/software. Finding the best parameters for each functions took quite a long time for me to experiment around without a fundamental knowledge background.


## How far I've got to
I have implemented (or selected or/and modified) 3 programs. These are 
1. Eye detection from webcam
2. Pupil detection from webcam (with Cascade Classifier( with 'haarcascade_eye.xml'), Morphology Extraction, contour funding in image, and blob detection)
3. Pupil detection from static eye image (using Hough Circle Transform from OpenCV)


## About the CODEs !

1. Webcam-Eye-Detect
==================

This script is modified from Webcam-Face-Detect from https://github.com/shantnu/Webcam-Face-Detect.git by shantnu.
It detects eye object and draw a square on each eye. The eye object model was trained and stored in haarcascade_eyes.xml. See https://github.com/Itseez/opencv/tree/master/data/haarcascades. 

------------------------------------------------------------------------------------------------

To run the program, type

python webcam.py haarcascade_eyes.xml


2. Pupil detect from webcam camera
====================================
This orginal script is from Pupil detect from https://gist.github.com/67c14af0d5afaae5b18c.git by edfungus

My main task here was modifying the code so it detects only pupil not an entire eye. I added blob detection function (using blobMaker from SimpleCV), find the best parameters and filter only the results that pupil is detected. 
Issues here still are that the image results saved include all the images, not just those that have a blob drawn on the pupil. 

------------------------------------------------------------------------------------------------

To run the program, type

python pupil.py

Note: that haarcascade_eye.xml is required in the working directory in order to make the script work
You can get the haarcascade file from this link: https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml


3. Pupil (and pupil only) detect from eye.jpg picture
===============================================================
Sorce of learning: http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_houghcircles/py_houghcircles.html

I modified the script and applied it to my eye.jpeg image. OpenCV uses Hough Transform to find circles in an image. Pupil is generally a circle. So, I tried to apply this method to find a pupil. However, the challenging part is to find the best parameter for cv2.HoughCircles(). I experimented adjusting param 1 and param 2 until it found a circle for a pupil. The bigger the param 2 is, the less number of circle detected. For the next step, we can create a simple GUI including for user to help inputing these parameters to find a pupil from diferent eye images.

------------------------------------------------------------------------------------------------

To run the program, type

python pupil_only.py


##Future Tasks
=================
- improve the Pupil detect from webcam camera program to 
	- save only the pupil results
	- store blob(pupil) data for future machine learning tasks to predict neurological condition
	- collect patients and control data eye video data
	- train/test machine learning model.