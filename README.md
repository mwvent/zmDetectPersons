# zmDetectPersons

Simple Python script to efficientley analyse Zoneminder event videos using OpenCV dnn and look for people in them
If a person is found ( or not ) a tag is added to the Event notes - HasPerson: 1 or HasPerson: 0 - this can be used to create filters to sepeate events caused by people from other noise

Events must have an mp4 video saved not just JPGs
Events will be ignored if they have no alarm frames 
The script only processes parts of the video with alarm frames to save processing time

The amount of frames per second of (alarmed) parts of the video to be processed is adjustable in the script by changing the readFPS var

Tested on Ubuntu 18.04

# Install steps

Create a python(3.6+) venv at /opt/zmDetectPersons
	pip Install Dependencies from dependencies.txt
	copy service.sh & zmDetectPersons.py to /opt/zmDetectPersons
	make /opt/zmDetectPersons/dnn-data folder
		download coco.names yolov3-tiny.cfg yolov3-tiny.weights in dnn-data folder
If using hardware OpenCL
	make an opencl cache folder ( i.e /var/opencl_cache )
	edit service.sh and edit OPENCV_OCL4DNN_CONFIG_PATH to point to it
	in zmDetectPersons.py edit dnn_target = cv.dnn.DNN_TARGET_CPU to dnn_target = cv.dnn.DNN_TARGET_OPENCL
Check /etc/zm/zm.conf database user and pass match those in zmDetectPersons.py

Now run service.sh as a user that has read permissions on video storage to check that the script runs
If everything is ok you may set debugOutput = False in zmDetectPersons.py to save logging

# To add as a permanent background service
Add zmDetectPersons.service to /etc/systemd/system/
Recommended but not essential add User=www-data to .service so as not to run as root. If using hardware OpenCL ensure cache folder and contents has read&write access by www-data
run systemctl enable zmDetectPersons.service
    service zmDetectPersons start

