#!/usr/bin/python
'''
    Script for fusing YOLOv4 with DeepSORT and acoustic data to compute fish swimming speed. 
    
'''
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import re
import datetime
import subprocess
from pylab import sort

from viewSavedData import loadFileNames
from src.fusePlot import parseVideoTime



def loadVideoFileNames_fusion(startTime, stopTime):
    videofiles = []
    hhmmss_list = []

    directory = os.getcwd()+'/deepsort/data/video'
    
    #hhmm = str(re.findall('[0-9]{2}-[0-9]{2}', filename))[2:-2]
    #hhmmss = str(re.findall('[0-9]{2}-[0-9]{2}-[0-9]{2}', filename))[2:-2]

    for root, dirs, filenames in os.walk(directory, topdown=False):
        for filename in filenames:
            if 'DS' in filename:
                continue
            hhmm = str(re.findall('[0-9]{2}-[0-9]{2}-[0-9]{2}', filename))[14:-5]
            hhmm = hhmm.replace("-", ":")
            if startTime <= hhmm <= stopTime:
                videofiles.append(root+'/'+filename)
                print("video added:", filename)

    videofiles = sort(videofiles) ## Sorting by time

    return videofiles






if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--start",
        action="store",
        type=str,
        default='12:04',
        dest='startTime',
        help="View data acquired from NN:NN o'clock",
    )
    parser.add_argument(
        "--stop",
        action="store",
        type=str,
        default='13:29',
        dest="stopTime",
        help="View data acquired until NN:NN o'clock",
    )

    
    args = parser.parse_args()
    
    
    rx_files = loadFileNames(args.startTime, args.stopTime, True, True)
    videoFiles = loadVideoFileNames_fusion(args.startTime, args.stopTime)
    
    print(videoFiles)
    
    for video in videoFiles:
       
        vidStart, vidEnd = parseVideoTime(video)
        print("Video timestamp [start, end]", vidStart, vidEnd)
        
        rxFilesInVideo = []
        rxFiles_timeStamps = []
        
        for rx_file in rx_files:
            hhmmssff = str(re.findall('[0-9]{2}:[0-9]{2}:[0-9]{2}.[0-9]{5}', rx_file))[2:-2]
            if vidStart <= hhmmssff <= vidEnd:
                ### Only capturing RX files acquired during current video ###
                rxFilesInVideo.append(rx_file)
                rxFile_timeStamp = str(re.findall('[0-9]{2}:[0-9]{2}:[0-9]{2}.?[0-9]{5}', rx_file))[2:-2]
                rxFiles_timeStamps.append(rxFile_timeStamp)#datetime.datetime.strptime(rxFile_timeStamp, '%H:%M:%S.%f'))
                print("added:", rxFile_timeStamp)
                continue
            
            elif hhmmssff > vidEnd:
                break
        print("video:", video)
        print("rx files:", rxFiles_timeStamps)
        rxFiles_timeStamps =str(rxFiles_timeStamps).strip('[]') 
        #print("wtf", rxFiles_timeStamps)
        #quit()
        DeepSORT_cmd = 'object_tracker.py --weights ./checkpoints/FISC_4 --video '+video+ ' --info --SF_timestamps ' + rxFiles_timeStamps
        test = '--weights ./checkpoints/FISC_4 --video '+video+ ' --info --SF_timestamps ' + rxFiles_timeStamps
        #quit()
        #object_tracker(--weights ./checkpoints/FISC_4 --video +video+ --info --SF_timestamps + rxFiles_timeStamps)
        subprocess.call(['python3', 'object_tracker.py', '--weights', './checkpoints/FISC_4', '--info', '--video', video, '--SF_timestamps', rxFiles_timeStamps], cwd=os.getcwd()+'/deepsort')
     
            
    

    
    
    
    
    
    
    
    
