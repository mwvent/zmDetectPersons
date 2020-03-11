import cv2 as cv
import numpy as np
import mysql.connector
import time
import re
import os
from threading import Thread
import datetime
import sys

# Config
class zmDetectPersonsOpts :
    logLevel = 10        # 0 nothing 1 errors 2 info 3-5 debug
    readFPS = 4          #Desired number of frames to analyze per second of video
    confThreshold = 0.22 #DNN Confidence threshold
    minZmFrameScore = 0  # Minimum `score` zoneminder has tagged frame - zero for all alarm frames
    zmMonitors = [ 11 , 12 ] # List of zm monitor ids to process on ( usually substreams )
    dbHost = "localhost" #MySql Zoneminder Database server
    dbUser = "zmuser"    #Mysql Zoneminder Database user ( TODO load from /etc/zm )
    dbPass = "zmpass"    #MySql Zoneminder Database pass ( TODO load from /etc/zm )
    dnn_modelConfiguration = "dnn-data/yolov3-tiny.cfg"
    dnn_modelWeights = "dnn-data/yolov3-tiny.weights"
    dnn_classesFile = "dnn-data/coco.names"
    useGPU = True
    useGPUandCPU = True
    useThreading = True
opts = zmDetectPersonsOpts()

# Standard Logging Funtion
def log( level, logTxt, newLine = True ) :
    global opts
    if opts.logLevel == 0 : return
    if not opts.logLevel >= level : return
    newLineTxt = "\n" if newLine else "           \r"
    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sys.stderr.write( datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " " + logTxt + newLineTxt)

# Container for opencv dnn net with other info needed for using it
class dnn_net :
    net = None
    outputsNames = None
    inpWidth = None
    inpHeight = None
    classes = None

# Container and initialiser for usable dnn nets
# Reset funtion is provided as sometimes on a long running process net.forward
# begins to throw exeptions - resetting fixes the problem
class dnn_nets :
    def __init__(self, opts) :
        self.opts = opts
        self.reset()
        
    def reset(self) :
        self.nets = []
        opts = self.opts
        self.inpWidth = 416 # TODO READ FROM FILE opts.dnn_modelConfiguration
        self.inpHeight = 416 # TODO READ FROM FILE opts.dnn_modelConfiguration
        with open(self.opts.dnn_classesFile, 'rt') as f:
                self.classes = f.read().rstrip('\n').split('\n')
        if self.opts.useGPU or self.opts.useGPUandCPU :
            os.environ["OPENCV_OCL4DNN_CONFIG_PATH"] = "/var/opencl_cache/" # TODO Use Config
            os.environ["OPENCV_OPENCL_DEVICE"] = ":GPU:0" # TODO Use Config
            newNet = dnn_net()
            newNet.classes = self.classes
            newNet.net = cv.dnn.readNetFromDarknet(opts.dnn_modelConfiguration, opts.dnn_modelWeights)
            newNet.net.setPreferableBackend(cv.dnn.DNN_BACKEND_DEFAULT)
            newNet.net.setPreferableTarget(cv.dnn.DNN_TARGET_OPENCL)
            netLayersNames = newNet.net.getLayerNames()
            newNet.outputsNames = [netLayersNames[i[0] - 1] for i in newNet.net.getUnconnectedOutLayers()]
            self.nets.append(newNet)
        if not self.opts.useGPU or self.opts.useGPUandCPU :
            newNet = dnn_net()
            newNet.classes = self.classes
            newNet.net = cv.dnn.readNetFromDarknet(opts.dnn_modelConfiguration, opts.dnn_modelWeights)
            newNet.net.setPreferableBackend(cv.dnn.DNN_BACKEND_DEFAULT)
            newNet.net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
            netLayersNames = newNet.net.getLayerNames()
            newNet.outputsNames = [netLayersNames[i[0] - 1] for i in newNet.net.getUnconnectedOutLayers()]
            self.nets.append(newNet)
        
    def getNet(self, index) :
        return self.nets[index]
    
    def get(self) :
        return self.nets
        
    def count(self) :
        return len(self.nets)

# Wraps the zoneminder database and provided an interface to the queries used
class zmDBC :
    def __init__(self, opts) :
        self.opts = opts
        self.db = mysql.connector.connect(
            host=opts.dbHost, user=opts.dbUser, passwd=opts.dbPass, database="zm", autocommit=True
        )
    def workSQL(self, excludeEvents, workLeft = False) :
        SQL_SELECT_NEXTEVENT = [ "Events.Id AS EventId",
            "Concat( Path, '/', MonitorId, '/', Date(StartTime), '/', Events.Id ) AS Path",
            "Events.Notes AS CurrentNotes"
        ]
        SQL_SELECT_WORKLEFT = [ "count(Events.Id)", "sum(Events.AlarmFrames)" ]
        SQL_FROM = "Events INNER JOIN Storage ON StorageId=Storage.Id"
        SQL_CONDITIONS = [ 
            "Notes NOT LIKE '%HasPerson:%'",
            "EndTime IS NOT NULL AND EndTime < (NOW() - INTERVAL 1 MINUTE)",
            "StartTime > (NOW() - INTERVAL 48 HOUR)",
            "AlarmFrames > 0"
        ]
        for eventId in excludeEvents:
            SQL_CONDITIONS.append("Events.Id<>"+str(eventId))
        if len(opts.zmMonitors) > 0 :
            SQL_CONDITIONS_MONS = []
            for monId in opts.zmMonitors :
                SQL_CONDITIONS_MONS.append("MonitorId = " + str(monId))
            SQL_CONDITIONS.append( "(" + " OR ".join(SQL_CONDITIONS_MONS) + ")" )
        SQL_END_NEXTEVENT = "ORDER BY StartTime DESC LIMIT 1"
        SQL_END_WORKLEFT = ""
        SQL_NEXTEVENT = "SELECT " + ", ".join(SQL_SELECT_NEXTEVENT) + " FROM " + \
            SQL_FROM + " WHERE " + " AND ".join(SQL_CONDITIONS) + " " + SQL_END_NEXTEVENT + ";"
        SQL_WORKLEFT = "SELECT " + ", ".join(SQL_SELECT_WORKLEFT) + " FROM " + \
            SQL_FROM + " WHERE " + " AND ".join(SQL_CONDITIONS) + " " + SQL_END_WORKLEFT + ";"
        return SQL_NEXTEVENT if not workLeft else SQL_NEXTEVENT
        
    def readNextEvent(self, excludeEvents = []) :
        SQL = self.workSQL(excludeEvents, False)
        dbcursor = self.db.cursor()
        dbcursor.execute(SQL)
        dbresult = dbcursor.fetchall()
        dbcursor.close()
        return dbresult
        
    def readFramesForEvent(self, eventid) :
        SQL = "SELECT Delta FROM Frames WHERE EventId=%s AND Type='Alarm' AND Score >= %s;"
        VARS = ( eventid, self.opts.minZmFrameScore )
        dbcursor = self.db.cursor()
        dbcursor.execute(SQL, VARS)
        alarmFrameTimesRaw = dbcursor.fetchall()
        dbcursor.close()
        alarmFrameTimes = []
        for alarmSecond in alarmFrameTimesRaw:
            alarmFrameTimes.append(float(alarmSecond[0]))
        return alarmFrameTimes
    
    def readWorkLeft(self, excludeEvents = []) :
        SQL = self.workSQL(excludeEvents, True)
        dbcursor = self.db.cursor()
        dbcursor.execute(SQL)
        dbresult = dbcursor.fetchall()
        dbcursor.close()
        return dbresult
        
    def updateEventNotes(self, eventid, newNotes) :
        SQL="UPDATE Events SET Notes=%s WHERE Id=%s;"
        VARS=(newNotes, eventid)
        dbcursor = self.db.cursor()
        dbcursor.execute(SQL,VARS)
        dbcursor.close()

# Storage and loading/processing funtions for a zoneminder event
class zmevent :
    eventId = None # The Id of the loaded event
    frames = None # Array of 4D blobs from the events loaded alarm frames
    framesprocessed = None # Array of status of each frame, 0=Loaded 1=Processing 2=Processed
    hasPerson = None # True/False if person found or None if processing not done yet
    finishedProcessing = False # Flagged True once event Loading/Processing/Saving complete
    
    # Open empty event - optionally caller provides its persistent database connection
    def __init__( self, opts, nets, db = None ) :
        self.opts = opts
        self.nets = nets
        if db is None :
            self.db = zmDBC(opts)
        else :
            self.db = db
    
    # Attempt to load event information and its frames from unprocessed events in the database
    # if caller has already loaded other events it must provide a list of eventIds to
    # exlude from the database fetch
    def loadNewEvent( self , excludeEvents = [] ) :
        assert( self.eventId is None )
        dbresult = self.db.readNextEvent( excludeEvents )
        if not len(dbresult) > 0 :
            return
        self.eventId = str(dbresult[0][0])
        self.videoPath = dbresult[0][1] + "/" + self.eventId + "-video.mp4"
        self.snapshotPath = dbresult[0][1] +"/" + "snapshot.jpg"
        self.currentnotes = dbresult[0][2]
        self.alarmFrameTimes = self.db.readFramesForEvent(self.eventId)
        # self.readFramesFromFile(self.snapshotPath, 0)
        self.readFramesFromFile(self.videoPath, opts.readFPS)
        log(4, "Event:" + str(self.eventId) + " Loaded " + str(len(self.frames)) + " frames" )
    
    # return True if database fetch yeilded a result
    def hasEvent(self) :
        return not self.eventId is None
    
    # return True if an event was loaded and is not yet processed
    def hasUnprocessedEvent(self) :
        return self.hasPerson is None and not self.hasEvent() is None

    # return human readable stats for event 
    def getEventStats(self) :
        if self.eventId is None :
            return "Empty Event"
        elif not self.finishedProcessing and self.framesprocessed.count(2) == 0:
            return  "Event:" + str(self.eventId) + \
                    " Frames:" + str(len(self.frames))
        elif not self.finishedProcessing :
            return  "Event:" + str(self.eventId) + \
                    " Processed:" + str(self.framesprocessed.count(2)) + \
                    "/" + str(len(self.frames))
        else :
            return  "Event:" + str(self.eventId) +  \
                    " Person:" + str(self.hasPerson) + \
                    " Frames:" + str(self.framesprocessed.count(2)) + str(self.netProcessedCounts) + \
                    " Conf:" + str(self.maxPersonConfidence) + \
                    " Time:" + str(self.timetaken) + \
                    " FPS:" + str(self.fps)
    
    # Load frames from a video or image into the event
    def readFramesFromFile( self, path, readFPS ) :
        assert( not self.hasEvent() is None )
        # open frame container
        cap = cv.VideoCapture(path, cv.CAP_FFMPEG)
        # frame Skip calc
        frameSkipCounter = 0
        videoFPS = int(cap.get(cv.CAP_PROP_FPS))
        videoFrameLenS = 1/cap.get(cv.CAP_PROP_FPS) if videoFPS > 0 else 1
        if readFPS > videoFPS or readFPS == videoFPS :
            frameSkipAmount = 1
        elif videoFPS == 0 or readFPS == 0 :
            frameSkipAmount = 0
        else :
            frameSkipAmount = round( videoFPS / readFPS )
        # loop & gather all frames valid for processing
        self.frames = []
        self.framesprocessed = []
        readingVideoFrames = True
        while readingVideoFrames :
            hasFrame, rawframe = cap.read()
            # if reading has finished then return any frame(s)
            if not hasFrame : break
            # ignore frames more than 1 frame duration away from a zoneminder alarm frame
            currentTime = float(cap.get(cv.CAP_PROP_POS_MSEC)/1000)
            foundCloseFrame = False if readFPS > 0 else True
            for alarmFrameTime in self.alarmFrameTimes :
                minTime = ( alarmFrameTime - videoFrameLenS )
                maxTime = ( alarmFrameTime + videoFrameLenS )
                if currentTime >= minTime and currentTime <= maxTime :
                    foundCloseFrame = True
                    break
            if not foundCloseFrame :
                frameSkipCounter = 0
                continue
            # frameskip to only read desired fps
            frameSkipCounter -= 1
            if frameSkipCounter < 1 :
                frameSkipCounter = frameSkipAmount
            else :
                continue
            # Finally have a valid frame to store ( as a 4D blob ready for dnn forward )
            blob = cv.dnn.blobFromImage(
                rawframe.copy(), 1/255,
                (self.nets.inpWidth, self.nets.inpHeight),
                [0,0,0], swapRB=False, crop=False
            )
            self.frames.append(blob)
            self.framesprocessed.append(0) # mark the stored blob as unprocessed
    
    # Get the confidene a blob contains a person
    def checkFrameForPerson( self, frameIndex, net, returnArr, returnIndex) :
        try :
            maxPersonConfidence = 0
            net.net.setInput(self.frames[frameIndex])
            outs = net.net.forward(net.outputsNames)
            for out in outs :
                for detection in out:
                    scores = detection[5:]
                    classId = np.argmax(scores)
                    confidence = scores[classId]
                    if net.classes[classId] == "person":
                        maxPersonConfidence = confidence if confidence > maxPersonConfidence else maxPersonConfidence
            returnArr[returnIndex] = maxPersonConfidence
            self.framesprocessed[frameIndex] = 2
        except Exception as e:
            log(1, "Error in checkFrameForPerson " + str(e))
            returnArr[returnIndex] = -2
            self.framesprocessed[frameIndex] = 0
    
    # Does what it says
    def getUnprocessedFrameCount( self ) :
        return self.framesprocessed.count(0) + self.framesprocessed.count(1)

    # Pass all frames to checkFrameForPerson until out of frames or
    # confidence a person was found > threshold set in options
    # If mutiple dnn's are avaible then use 'em all 
    # Threaded if set in options to use dnns in parallel or if 
    # not threading process one on each availible dnn in turn
    def checkFramesForPerson( self ) :
        processingStartTime = time.time()
        self.timetaken = 0
        self.fps = 0
        self.maxPersonConfidence = 0
        # make a thread for each net
        if self.opts.useThreading and self.nets.count() > 1:
            threads = np.full(self.nets.count(), Thread(), dtype=object)
            threadReturnVals = np.full(len(threads), -1, dtype=np.float64)
        self.netProcessedCounts = np.full(self.nets.count(), 0, dtype=np.int)
        while (self.getUnprocessedFrameCount() > 0) and not self.hasPerson :
            for netIndex, net in enumerate(self.nets.get()) :
                if self.opts.useThreading and self.nets.count() > 1:
                    # thread for net is still running
                    if threads[netIndex].isAlive() :
                        continue
                    # if thread has returned a valid result
                    if threadReturnVals[netIndex] > -1 :
                        confFound = threadReturnVals[netIndex]
                        threadReturnVals[netIndex] = -1
                        self.netProcessedCounts[netIndex] += 1
                        framesProcessed += 1
                        if confFound >= self.maxPersonConfidence :
                            self.maxPersonConfidence = confFound
                        self.hasPerson = True if confFound >= opts.confThreshold else False
                        if self.hasPerson : break
                        continue
                    # if thread returned an error
                    if threadReturnVals[netIndex] == -2 :
                        log(1, "Had an issue processing frame for net " + str(netIndex) + " attempting net restart")
                        nets.reset()
                        threadReturnVals[netIndex] = -1
                        continue
                    # thread is not running - give it some work if frames left
                    if self.framesprocessed.count(0) > 0 :
                        frameIndex = self.framesprocessed.index(0)
                        self.framesprocessed[frameIndex] = 1
                        threads[netIndex] = Thread(
                            target = self.checkFrameForPerson,
                            args = ( frameIndex, net, threadReturnVals, netIndex )
                        )
                        threads[netIndex].start()
                        continue
                else :
                    # if not using threads or non threading option then use main thread
                    frameIndex = self.framesprocessed.index(0)
                    retArr = [-1]
                    self.checkFrameForPerson(frameIndex, net, retArr, 0)
                    confFound = retArr[0]
                    self.netProcessedCounts[netIndex] += 1
                    if confFound >= self.maxPersonConfidence :
                        self.maxPersonConfidence = confFound
                    self.hasPerson = True if confFound >= opts.confThreshold else False
                    if self.hasPerson : break
            # Wrap up and save stats
            self.timetaken = round(time.time() - processingStartTime,2)
            framesProcessed = self.framesprocessed.count(2)
            self.fps = round(framesProcessed/(time.time() - processingStartTime),2)
            self.maxPersonConfidence=round(self.maxPersonConfidence,2)
    
    # Process loaded frames - save person found info to zm database - log
    def process( self ) :
        assert( self.hasPerson is None )
        self.checkFramesForPerson()
        newNotes = "HasPerson: 1" if self.hasPerson else "HasPerson: 0"
        newNotes = self.currentnotes + " " + newNotes + " (" + str(self.maxPersonConfidence) + ")"
        self.db.updateEventNotes( self.eventId, newNotes )
        self.finishedProcessing = True
        log(2, self.getEventStats())


# Main Loop - Non Threading
if not opts.useThreading :
    hadWork = True
    nets = dnn_nets(opts)
    db = zmDBC(opts)
    while True :
        event = zmevent( opts, nets, db )
        event.loadNewEvent()
        if not event.hasUnprocessedEvent() and hadWork:
            hadWork = False
            log(2, "No Work Left")
        if not event.hasUnprocessedEvent() :
            time.sleep(1)
            continue
        hadWork = True
        event.process()

# Main Loop - With Threading
# Attempt to load events and process in parallel - useful when
# lots of small events are being created rapidly ( i.e rain and wind triggers )
loadThread = Thread() # Thread for finding next event and loading its frames
loadThreadDB = zmDBC(opts) # Peristent database connection for use by loadThread
loadingEvent = None # Store loadThreads working event object - main thread pulls once finished
processThread = Thread() # Thread for processing event frames &saving result
processThreadDB = zmDBC(opts) # Peristent database connection for use by processThread
processingEvent = None # Store processThreads working event object
processingQue = [] # main thread pushes event objects completed by processThread into here
                   # & main thread pulls event objects from here for the processThread
processingQue_maxsize = 5 # how many events to load before waiting for processThread to finish

nets = dnn_nets(opts) # Persistent loaded opencv dnn nets to pass to threads
hadWork = True # Flag used to ensure the "No Work Left" Message is shown just once when no new work
while True :
    noNewWorkFound = False # Flag to be set if loadingThread returns with no work
    if not loadThread.isAlive() :
        # If loadThread has returned with some work move it to the que
        if not loadingEvent is None :
            if not loadingEvent.eventId is None :
                loadingEvent.db = processThreadDB
                processingQue.append(loadingEvent)
                loadingEvent = None
            else :
                noNewWorkFound = True # Flag loadingThread returned with no work
        # loadThread is clear - start it again to check for new work
        if not len(processingQue) >= processingQue_maxsize :
            excludeEvents = []
            for event in processingQue :
                excludeEvents.append(event.eventId)
            if not processingEvent is None :
                excludeEvents.append(processingEvent.eventId)
            loadingEvent = zmevent( opts, nets, loadThreadDB )
            loadThread = Thread(
                target = loadingEvent.loadNewEvent, 
                 args = [excludeEvents]
            )
            loadThread.start()
    
    # if the processing thread is not running and work is sitting in the que then
    # move some work out of the que and start the thread running on it
    # do not need to check result of any previous processing result so just overwrite it
    if not len(processingQue) == 0 and not processThread.isAlive() :
        processingEvent = processingQue.pop(0)
        processThread = Thread( target = processingEvent.process )
        processThread.start()
        hadWork = True
    
    # if no new work is has come from db or the que is full then sit and wait for the processing
    # thread to finish. This ensures not wasting CPU on running the main loop rapidly without work
    if processThread.isAlive() and ( noNewWorkFound or len(processingQue) >= processingQue_maxsize ):
        processThread.join()

    # If everything is loaded and processed then sleep a second before starting again
    # Prevents CPU being eaten by idle main loop
    if not processThread.isAlive() and noNewWorkFound and len(processingQue) == 0:
        if hadWork :
            log(2, "No Work Left")
            hadWork = False
        time.sleep(1)


    
