import cv2 as cv
import numpy as np
import mysql.connector
import time

# Config
debugOutput = True
readFPS = 20          #Desired number of frames to analyze per second of video
confThreshold = 0.22  #DNN Confidence threshold
minZmFrameScore = 4  # Minimum `score` zoneminder has tagged frame - zero for all alarm frames
inpWidth = 416       #DNN Width of network's input image
inpHeight = 416      #DNN Height of network's input image
dnn_target = cv.dnn.DNN_TARGET_OPENCL # cv.dnn.DNN_TARGET_CPU / cv.dnn.DNN_TARGET_OPENCL
classesFile = "dnn-data/coco.names"
modelConfiguration = "dnn-data/yolov3-tiny.cfg"
modelWeights = "dnn-data/yolov3-tiny.weights"
dbHost = "localhost" #MySql Zoneminder Database server
dbUser = "zmuser"    #Mysql Zoneminder Database user ( TODO load from /etc/zm )
dbPass = "zmpass"    #MySql Zoneminder Database pass ( TODO load from /etc/zm )
# Set nvidia_maxtemp to a non 0 value to monitor nvidia-smi gpu temp and switch to CPU if overtemp
# This is Niche use case but unfortunatley needed on my old 560SE that can creep > 100C and cause poweroff!
# will most likely need tweaks if there is more than one nvidia gpu on board as nvidia-smi will return > 1 temp
nvidia_maxtemp = 85

# libs that are only required for debug ooutput
if debugOutput :
    import datetime
    import sys
if nvidia_maxtemp > 0 :
    import os

# Setup dnn stuff
classes = None
with open(classesFile, 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')
if nvidia_maxtemp > 0 and dnn_target == cv.dnn.DNN_TARGET_OPENCL :
    netCPU = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
    netCPU.setPreferableBackend(cv.dnn.DNN_BACKEND_DEFAULT)
    netCPU.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
    netGPU = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
    netGPU.setPreferableBackend(cv.dnn.DNN_BACKEND_DEFAULT)
    netGPU.setPreferableTarget(cv.dnn.DNN_TARGET_OPENCL)
    net = netGPU
else :
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
if nvidia_maxtemp > 0 :
    nvidia_maxtemp_isPaused = False
havevideo=False
framesread=0
frameSkipCounter=0
frameSkipAmount=1

# Main Loop
while True:
    # Throttle Down On Nvidia Max Temp see config 
    if nvidia_maxtemp > 0 :
        temp=int(os.popen("nvidia-smi -q -d temperature | grep \"GPU Current\" | cut -d \":\" -f 2 | cut -d \" \" -f2").read())
        if temp > nvidia_maxtemp and not nvidia_maxtemp_isPaused :
            nvidia_maxtemp_isPaused = True
            if debugOutput :
                timeS = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                sys.stderr.write(timeS + " GPU TEMP TOO HIGH - SWITCHING TO CPU\n")
        if temp < ( nvidia_maxtemp - 4 )and nvidia_maxtemp_isPaused :
            nvidia_maxtemp_isPaused = False
            if debugOutput :
                timeS = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                sys.stderr.write(timeS + " GPU Temp recovered - resume GPU\n")
        if nvidia_maxtemp_isPaused :
            net = netCPU
        else :
            net = netGPU

    # If not processing video Get next event to work on
    if not havevideo:
        SQL_SELECT_NEXTEVENT = [ "Events.Id AS EventId",
            "Concat( Path, '/', MonitorId, '/', Date(StartTime), '/', Events.Id, '/', Events.Id , '-video.mp4' ) AS Path",
            "Events.Notes AS CurrentNotes"
        ]
        SQL_SELECT_WORKLEFT = [ "count(Events.Id)", "sum(Events.AlarmFrames)" ]
        SQL_FROM = "Events INNER JOIN Storage ON StorageId=Storage.Id"
        SQL_CONDITIONS = [ "Notes NOT LIKE '%HasPerson:%'",
            "EndTime IS NOT NULL AND EndTime < (NOW() - INTERVAL 1 MINUTE)",
            "StartTime > (NOW() - INTERVAL 48 HOUR)",
            "AlarmFrames > 0"
        ]
        SQL_END_NEXTEVENT = "ORDER BY StartTime DESC LIMIT 1"
        SQL_END_WORKLEFT = ""

        SQL_NEXTEVENT = "SELECT " + ", ".join(SQL_SELECT_NEXTEVENT) + " FROM " + \
            SQL_FROM + " WHERE " + " AND ".join(SQL_CONDITIONS) + " " + SQL_END_NEXTEVENT + ";"

        SQL_WORKLEFT = "SELECT " + ", ".join(SQL_SELECT_WORKLEFT) + " FROM " + \
            SQL_FROM + " WHERE " + " AND ".join(SQL_CONDITIONS) + " " + SQL_END_WORKLEFT + ";"

        # Get count of work remaining
        if debugOutput :
            db.commit()
            dbcursor = db.cursor()
            dbcursor.execute(SQL_WORKLEFT)
            dbresult = dbcursor.fetchall()
            dbcursor.close()
            if len(dbresult) > 0 :
                if int(dbresult[0][0]) > 0 :
                    eventsToProcess = str(dbresult[0][0])
                    framesToProcess = str(dbresult[0][1])
                    timeS = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    sys.stderr.write(timeS + " Have " + eventsToProcess + " events left with " + \
                        framesToProcess + " alarmed frames\n")

        # Get next video to process
        db.commit()
        dbcursor = db.cursor()
        dbcursor.execute(SQL_NEXTEVENT)
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
        # Get the alarm frames with a score high enougth put them into usable array
        SQL = "SELECT Delta FROM Frames WHERE EventId=%s AND Type='Alarm' AND Score >= %s;"
        VARS = ( eventid, minZmFrameScore )
        dbcursor = db.cursor()
        dbcursor.execute(SQL, VARS)
        alarmFrameTimesRaw = dbcursor.fetchall()
        dbcursor.close()
        alarmFrameTimes = []
        for alarmSecond in alarmFrameTimesRaw:
            alarmFrameTimes.append(float(alarmSecond[0]))
        cap = cv.VideoCapture(videopath, cv.CAP_FFMPEG)
        # init stats
        framesread = 0
        framesskipped = 0
        havevideo = True
        personDetected = False
        maxPersonConfidence = 0
        # frame Skip calc
        videoFPS = int(cap.get(cv.CAP_PROP_FPS))
        videoFrameLenS = 1/cap.get(cv.CAP_PROP_FPS)
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
            logTxt = timeS + " Starting"
            logTxt+= " Event=" + eventid
            logTxt+= " Video=" + videopath
            logTxt+= " AlarmFrames>" + str(minZmFrameScore) +  "=" + str(len(alarmFrameTimes))
            logTxt+= " FPS=" + str(videoFPS)
            logTxt+= " FrameSkip=" + str(frameSkipAmount)
            sys.stderr.write(logTxt + "\n")
        continue

    # get next frame from the video
    hasFrame, rawframe = cap.read()
    if debugOutput :
        hadVideo = True

    # If reached end of video and not found a person log it to the database
    if not hasFrame:
        timeS = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if ( framesread + framesskipped ) == 0:
            if debugOutput :
                sys.stderr.write(timeS+" Processed no frames - score too low or corrupt video\n")
            newNotes = currentnotes + " HasPerson: -1"
        else:
            if debugOutput :
                timeTaken = str(time.process_time() - processingStartTime)
                logTxt = timeS + " Nobody found - Processed " + str(framesread)
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
    # zoneminders recorded frame times ( Delta column ) is unlikley to match
    # exactly the video position from CAP_PROP_POS_MSEC so we check if 
    # CAP_PROP_POS_MSEC is within 1 frame duration away from a Delta value
    currentTime = float(cap.get(cv.CAP_PROP_POS_MSEC)/1000)
    foundCloseFrame = False
    for alarmFrameTime in alarmFrameTimes :
        if currentTime >= ( alarmFrameTime - videoFrameLenS ) and currentTime <= ( alarmFrameTime + videoFrameLenS ) :
            foundCloseFrame = True
            break
    if not foundCloseFrame :
        framesskipped = framesskipped + 1
        # always reset framskip when there is a gap
        frameSkipCounter = 0
        continue

    # frameskip to only read desired fps
    frameSkipCounter = frameSkipCounter - 1
    if frameSkipCounter < 1 :
        frameSkipCounter = frameSkipAmount
        # reached 0 carry on and process frame
    else :
        # otherwise skip the frame
        framesskipped = framesskipped + 1
        continue

    framesread=framesread+1
    if debugOutput :
        timeTaken = time.process_time() - processingStartTime
        framesPerSec = framesread/timeTaken # 2 4 
        sys.stderr.write("Procesing frame " + str(framesread) + \
                " skipped " + str(framesskipped) +" FPS " + str(round(framesPerSec,2))  + "    \r")

    # Take a copy of the image - processing on the original can cause issues
    frame=rawframe.copy()

    # Magic copied and pasted dnn stuff that I dont understand
    # Create a 4D blob from a frame.
    # Suggested from one tutorial - missed people
    #blob = cv.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)
    # Seem ok 
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

