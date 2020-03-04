import cv2 as cv
import numpy as np
import mysql.connector
import time

# Config
debugOutput = True
readFPS = 6          #Desired number of frames to analyze per second of video
confThreshold = 0.22  #DNN Confidence threshold
inpWidth = 416       #DNN Width of network's input image
inpHeight = 416      #DNN Height of network's input image
dnn_target = cv.dnn.DNN_TARGET_CPU # cv.dnn.DNN_TARGET_CPU / cv.dnn.DNN_TARGET_OPENCL
classesFile = "dnn-data/coco.names"
modelConfiguration = "dnn-data/yolov3-tiny.cfg"
modelWeights = "dnn-data/yolov3-tiny.weights"
dbHost = "localhost" #MySql Zoneminder Database server
dbUser = "zmuser"    #Mysql Zoneminder Database user ( TODO load from /etc/zm )
dbPass = "zmpass"    #MySql Zoneminder Database pass ( TODO load from /etc/zm )

# libs that are only required for debug ooutput
if debugOutput :
    import datetime
    import sys

# Setup dnn stuff
classes = None
with open(classesFile, 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')
net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_DEFAULT)
net.setPreferableTarget(dnn_target)
netLayersNames = net.getLayerNames()
netOutputsNames = [netLayersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Connect to ZM Database
db = mysql.connector.connect(host=dbHost, user=dbUser, passwd=dbPass, database="zm", autocommit=True)
dbcursor = db.cursor()

# Init main loop
if debugOutput :
    hadVideo = True
havevideo=False
framesread=0
frameSkipCounter=0
frameSkipAmount=1

# Main Loop
while True:
    if not havevideo:
        # Get Next Video
        SQL = """ SELECT
                 Events.Id AS EventId,
                 Concat( Path, '/', MonitorId, '/', Date(StartTime), '/', Events.Id, '/', Events.Id , '-video.mp4' ) AS Path,
                 Events.Notes AS CurrentNotes
             FROM Events 
                 INNER JOIN Storage ON StorageId=Storage.Id 
             WHERE
                 Notes NOT LIKE '%HasPerson:%' 
                 AND EndTime IS NOT NULL AND EndTime < (NOW() - INTERVAL 1 MINUTE)
                 AND StartTime > (NOW() - INTERVAL 4 HOUR)
                 AND AlarmFrames > 0
             ORDER BY
                 StartTime DESC LIMIT 1;"""
        # Get next video to process
        dbcursor = db.cursor()
        dbcursor.execute(SQL)
        dbresult = dbcursor.fetchall()
        dbcursor.close()
        if len(dbresult) == 0:
            if debugOutput :
                if hadVideo :
                    timeS = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    sys.stderr.write(timeS + " No work - waiting\n")
                    hadVideo = False
            time.sleep(5)
            continue
        eventid = str(dbresult[0][0])
        videopath = dbresult[0][1]
        currentnotes = dbresult[0][2]
        # Get the Alarm Frame times for video - accuarate to 1s
        SQL = "SELECT round(Delta) FROM Frames WHERE EventId=%s AND Type='Alarm' GROUP BY round(Delta);"
        # SQL = "SELECT Delta FROM Frames WHERE EventId=%s AND Type='Alarm' ORDER BY Delta DESC;"
        VARS = ( eventid, )
        dbcursor = db.cursor()
        dbcursor.execute(SQL, VARS)
        alarmFrameTimesRaw = dbcursor.fetchall()
        dbcursor.close()
        alarmFrameTimes = []
        for alarmSecond in alarmFrameTimesRaw:
            alarmFrameTimes.append(int(alarmSecond[0]))
        cap = cv.VideoCapture(videopath, cv.CAP_FFMPEG)
        # init stats
        framesread = 0
        framesskipped = 0
        havevideo = True
        personDetected = False
        maxPersonConfidence = 0
        # frame Skip calc
        videoFPS = int(cap.get(cv.CAP_PROP_FPS))
        if readFPS > videoFPS or readFPS == videoFPS :
            frameSkipAmount = 1
        elif videoFPS == 0 or readFPS == 0 :
            frameSkipAmount = 0
        else :
            frameSkipAmount = round( videoFPS / readFPS )
        # log started
        if debugOutput :
            processingStartTime = time.process_time()
            timeS = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            logTxt = timeS + " Analyse event " + eventid
            logTxt = logTxt + " " + videopath + " with " + str(len(alarmFrameTimes))
            logTxt = logTxt + "s of alarm frames\n"
            sys.stderr.write(logTxt)
        continue

    # get frame from the video
    hasFrame, rawframe = cap.read()
    if debugOutput :
        hadVideo = True

    # If reached end of video and not found a person log it to the database
    if not hasFrame:
        timeS = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if ( framesread + framesskipped ) == 0:
            if debugOutput :
                sys.stderr.write(timeS+" Processed NO FRAMES no result ( corrupt video? )\n")
            newNotes = currentnotes + " HasPerson: Err"
        else:
            if debugOutput :
                timeTaken = str(time.process_time() - processingStartTime)
                logTxt = timeS + " Processed " + str(framesread)
                logTxt = logTxt + " frames skipped " + str(framesskipped)
                logTxt = logTxt + " max confidence " + str(maxPersonConfidence) 
                logTxt = logTxt + " time taken " + timeTaken  + "\n"
                sys.stderr.write( logTxt )
            newNotes = currentnotes + " HasPerson: 0"
        SQL="UPDATE Events SET Notes=%s WHERE Id=%s;"
        VARS=(newNotes, eventid)
        dbcursor = db.cursor()
        dbcursor.execute(SQL,VARS)
        db.commit()
        dbcursor.close()
        havevideo=False
        cap.release()
        continue

    # ignore non alarm frames
    currentTimeS = int(round(cap.get(cv.CAP_PROP_POS_MSEC)/1000))
    if alarmFrameTimes.count(currentTimeS) < 1 :
        framesskipped = framesskipped + 1
        continue

    # frameskip to read desired fps
    frameSkipCounter = frameSkipCounter - 1
    if frameSkipCounter < 1 :
        frameSkipCounter = frameSkipAmount
        # reached 0 carry on and process frame
    else :
        # otherwise skip the frame
        framesskipped = framesskipped + 1
        continue

    # Take a copy of the image - processing on the original can cause issues
    framesread=framesread+1
    frame=rawframe.copy()

    # Magic copied and pasted dnn stuff that I dont understand
    # Create a 4D blob from a frame.
    #blob = cv.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)
    blob = cv.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), swapRB=True, crop=False)
    # Sets the input to the network
    net.setInput(blob)
    # Runs the forward pass to get output of the output layers
    outs = net.forward(netOutputsNames)
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold :
                if classes[classId] == "person":
                    personDetected = True
                    if debugOutput and confidence > maxPersonConfidence:
                        maxPersonConfidence = confidence

    # Found a person - tag the video and stop processing
    if personDetected :
        newNotes = currentnotes + " HasPerson: 1"
        SQL="UPDATE Events SET Notes=%s WHERE Id=%s;"
        VARS=(newNotes, eventid)
        dbcursor = db.cursor()
        dbcursor.execute(SQL,VARS)
        db.commit()
        dbcursor.close()
        cap.release()
        havevideo=False
        if debugOutput :
            timeTaken = str(time.process_time() - processingStartTime)
            timeS = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            confS = "conf: " + str(maxPersonConfidence)
            sys.stderr.write(timeS+" Found Person " + confS  + "- Processed " + str(framesread) + " frames " + " time taken " + timeTaken  + "\n")
        continue

