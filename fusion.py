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
import csv
import os
import scipy.signal

from viewSavedData import loadFileNames
from src.fusePlot import parseVideoTime
from tools.acousticProcessing import gen_mfilt, matchedFilter, TVG, normalizeData, peakDetect

def genTrackerData():
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

def loadVideoFileNames_fusion(startTime, stopTime):
    videofiles = []
    hhmmss_list = []

    directory = os.getcwd()+'/deepsort/data/video'

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

def getEchoData(timestamp):
    directory = os.getcwd() + '/data/SectorFocus/11-03-22'

    for root, dirs, files in os.walk(directory, topdown=False):
        for file in files:
            if timestamp in file:
                filepath = root+'/'+file
                print("current file", file)

                data=np.load(filepath, allow_pickle=True)
                acqInfo = data['header']
                fc = int(acqInfo[0])
                BW = int(acqInfo[1])
                pulseLength = float(acqInfo[2])
                fs = int(acqInfo[3])
                acqInfo = data['header']
                c = 1472.5 ## From profiler

                echoData = data['sectorData'][:]
                downSampleStep = int(acqInfo[6])
                nSamples = len(echoData)

                ### Acquisition constants ###
                SampleTime = nSamples*(1/fs)
                Range = c*SampleTime/2
                samplesPerPulse = int(fs*pulseLength)  # How many samples do we get per pulse length
                tVec = np.linspace(0, SampleTime, nSamples)
                tVecShort = tVec[0:len(tVec):downSampleStep] # Downsampled time vector for plotting
                rangeVec = np.linspace(0, Range, len(tVec))
                rangeVecShort = np.linspace(0, Range, len(tVecShort)).round(decimals=2)
                print("fc:", int(fc), "BW:", int(BW), "fs:", int(fs), \
            		"plen (us):", int(pulseLength*1e6), "range:", Range, "c:", c, "Downsample step:", downSampleStep)

                ## Passing raw data through matched filter and peak detector ##
                mfilt = gen_mfilt(fc, BW, pulseLength, fs)

                echoData_Env, _ = matchedFilter(echoData, echoData, mfilt, downSampleStep) ## Extracting matched filter envelope
                echoData_Env = TVG(echoData_Env, Range, c, fs) ## Adding digital TVG
                echoData_Env[0:samplesPerPulse] = 0 ## To remove tx pulse noise

                ## Normalizing data from 0 to 1 ##
                echoData_Env_n = normalizeData(echoData_Env)

                ## Using CA-CFAR to detect peaks ##
                CH1_peaks_idx, CH1_noise, CH1_detections, CH1_thresholdArr = peakDetect(echoData_Env_n, num_train=80, num_guard=10, rate_fa=0.3)
                ## Using
                peaks, _ = scipy.signal.find_peaks(CH1_detections, distance=20, height=0.4)
                ## Deteksjonene må også bli normalisert 0-1 ##

                fig, ax = plt.subplots(1)
                #ax.plot(CH1_detections, color='magenta')
                ax.plot(peaks, echoData_Env_n[peaks], "x", alpha=0.5)
                ax.plot(echoData_Env_n, color='black', alpha=0.5)
                #ax.plot(CH1_detections, color='red', alpha=0.5)
                print(peaks)
                #plt.plot(echoData_Env_n)
                #plt.plot(rangeVecShort[CH1_peaks_idx], CH1_detections[CH1_peaks_idx], 'rD')
                #plt.plot()
                plt.show()
                #quit()




def extractTrackedTargets(file):

    data = []
    timestamps = []
    counter = 0
    with open(file) as f:
        data_reader = csv.reader(f)
        for i, line in enumerate(data_reader):
            if i == 0:
                print("header:", line)
                continue
            ts = line[0][3:-2]

            ID = int(line[1])
            #print(line[2])
            xmin = int(line[2])
            ymin = int(line[3])
            xmax = int(line[4])
            ymax = int(line[5][:-1])

            data.append([ts, ID, xmin, ymin, xmax, ymax])
            #print(ts, ID, xmin, ymin, xmax, ymax)
        f.close()
    data = np.array((data))
    print("HBbih\n")
    timestampArr = data[:,0]

    _, idx, inv, count = np.unique(data[:,1], return_index=True, return_inverse=True, return_counts=True)

    ind = np.argsort(inv)
    spp = np.cumsum(count[:-1])
    trackedArray = np.split(data[ind, :], spp, axis=0)

    for ID in trackedArray:
        if len(ID) < 2:
            continue
        ## Sorting by timestamp to plot in sequence
        data_sorted = sorted(ID, key=lambda x: x[0])
        #print(data_sorted[0])
        #quit()

        ## Open RX files that matches

        for sample in ID:
            timestamp = sample[0]
            getEchoData(timestamp)
        plt.show()
        quit()

    #print(idx, inv)
    #print(ind)
    #print(result)

    #for timestamps in
    #print(np.unique(data[:,0], axis=0))
    #for timestamp in
    #print(test[1])


def loadTrackerData(startTime, stopTime):
    directory = os.getcwd()+'/deepsort'
    trackerFiles = []
    for root, dirs, filenames in os.walk(directory, topdown=False):
        for filename in filenames:
            if 'DS' in filename or not filename.endswith('.csv'):
                continue
            hhmm = str(re.findall('[0-9]{2}-[0-9]{2}-[0-9]{2}', filename))[14:-5]
            hhmm = hhmm.replace("-", ":")
            if startTime <= hhmm <= stopTime:
                trackerFiles.append(root+'/'+filename)
                #print("file added:", filename)

    trackerFiles = sort(trackerFiles) ## Sorting by time

    return trackerFiles


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
    parser.add_argument(
        "--p",
        action="store_true",
        dest="process",
        help="Process generated data",
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        dest="generate",
        help="Generate CSV files to extract DeepSORT information at RX sample instances.",
    )

    args = parser.parse_args()

    if args.generate:
        genTrackerData(args.startTime, args.stopTime)
    if args.process:
        files = loadTrackerData(args.startTime, args.stopTime)
        #for file in files:
        extractTrackedTargets(files[0])
