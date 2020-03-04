#!/bin/bash
cd "$(dirname "$0")"
source ./bin/activate
export OPENCV_OCL4DNN_CONFIG_PATH=/var/opencl_cache/
while : ; do
	nice -n 19 python zmDetectPersons.py
	sleep 5
done
