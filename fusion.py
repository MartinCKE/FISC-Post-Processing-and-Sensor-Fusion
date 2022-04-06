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
import cv2
import time

from viewSavedData import loadFileNames
from src.fusePlot import parseVideoTime
from tools.acousticProcessing import gen_mfilt, matchedFilter, TVG, normalizeData, peakDetect, processEcho

fusionList_ID = [19, 132, 2, 114, 115, 42, 145, 195, 123, 144, 98, 107, 162, 4, 113, 51, 108, 130, 153, 138, 65, 93, 91, 113]

FOV_X = 46.7
FOV_Y = 28.6

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
	''' Function for loading video files that were pre-selected for fusion purposes.
		Input: Start and End time (interval) of which videos to load.
		Output: Array of video file paths which match time interval.
	'''
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
	''' Function for reading RX files and returning processed echo data.
		Input: Timestamp of RX file.
		Output: Processed echo data.
	'''
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
				c = 1472.5 ## From profiler data

				echoData = data['sectorData'][:]
				downSampleStep = int(acqInfo[6])
				nSamples = len(echoData)

				### Acquisition constants ###
				SampleTime = nSamples*(1/fs)
				Range = c*SampleTime/2
				samplesPerPulse = int(fs*pulseLength)  # How many samples do we get per pulse length
				tVec = np.linspace(0, SampleTime, nSamples)
				tVecShort = tVec[0:len(tVec):downSampleStep] # Downsampled time vector for plotting
				plen_d = (c*pulseLength)/2
				rangeVec = np.linspace(-plen_d, Range, len(tVec))
				rangeVecShort = np.linspace(-plen_d, Range, len(tVecShort)).round(decimals=2)
				print("fc:", int(fc), "BW:", int(BW), "fs:", int(fs), \
					"plen (us):", int(pulseLength*1e6), "range:", Range, "c:", c, "Downsample step:", downSampleStep)

				## Process echo data and fetch matched filter envelope and peaks in signal ##
				echoEnvelope, peaks = processEcho(echoData, fc, BW, pulseLength, fs, samplesPerPulse)

				return echoEnvelope, peaks, rangeVec
				'''

				fig, ax = plt.subplots(1)
				#ax.plot(echoData_Env)
				#ax.plot(CH1_detections, color='blue', label='detections')
				ax.plot(echoEnvelope, label='data')
				#ax.plot(echoData, label='raw data')
				ax.plot(peaks, echoEnvelope[peaks], "x", alpha=0.5, label=file)
				#ax.plot(echoData_Env_n, color='black', alpha=0.5)
				#ax.plot(CH1_detections, color='red', alpha=0.5)
				#plt.plot(echoData_Env_n)
				#plt.plot(rangeVecShort[CH1_peaks_idx], CH1_detections[CH1_peaks_idx], 'rD')
				#plt.plot()
				plt.legend()
				plt.draw()
				plt.
				#quit()
				'''

def getFrame(videoFile, timeStamp):
	''' Function for fetching video frame with specific timestamp
	'''
	print("Get frame from video:", videoFile)
	print("Get frame with timestamp:", timeStamp)
	startTime, endTime = parseVideoTime(videoFile)
	print(startTime, endTime)
	timeStamp = datetime.datetime.strptime(timeStamp, '%H:%M:%S.%f')

	### Create a VideoCapture object ###
	cap = cv2.VideoCapture(videoFile)

	### Check if file opened successfully ###
	if (cap.isOpened() == False):
	  print("Unable to read camera feed")

	### Frame resolution is obtained ###
	frame_width = int(cap.get(3))
	frame_height = int(cap.get(4))
	#print("Width, Height:", frame_width, frame_height)


	### Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file. ##
	#out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

	#calc_timestamps = [0]
	fps = cap.get(cv2.CAP_PROP_FPS)

	#timestamps = [datetime.datetime.strptime(startTime, '%H:%M:%S')]
	timestamps = [startTime]


	i = 0
	#fig, axs = plt.subplots(1, figsize=(6,4*1))
	#move_figure(fig, 800, 0)
	while(True):
		ret, frame = cap.read() ## read a frame from video

		if ret == True:
			try:
				last_frame_timestamp = datetime.datetime.strptime(timestamps[-1], '%H:%M:%S.%f')
				curr_frame_timestamp = datetime.datetime.strptime(timestamps[-1], '%H:%M:%S.%f')+ datetime.timedelta(seconds=1/fps)
				timestamps.append(str(curr_frame_timestamp)[-15:])

			except ValueError:
				last_frame_timestamp = datetime.datetime.strptime(timestamps[-1], '%H:%M:%S')
				curr_frame_timestamp = datetime.datetime.strptime(timestamps[-1], '%H:%M:%S') + datetime.timedelta(seconds=1/fps)
				timestamps.append(str(curr_frame_timestamp)[-15:])

			if abs(timeStamp - last_frame_timestamp) < datetime.timedelta(milliseconds=20) or (timeStamp - last_frame_timestamp).days == -1:
				print("rx file:", timeStamp)
				print("MATCH")
				return frame, last_frame_timestamp
				#frame_s = cv2.resize(frame, None, fx=0.7, fy=0.7, interpolation=cv2.INTER_AREA)
				#cv2.imshow(str(curr_frame_timestamp), frame_s)
				#cv2.waitKey(0)
			#frame_s = cv2.resize(frame, None, fx=0.7, fy=0.7, interpolation=cv2.INTER_AREA)
			#cv2.imshow(str(curr_frame_timestamp), frame_s)
			#cv2.waitKey(0)
			#if cv2.waitKey(33) & 0xFF == ord('q'):
			#	continue
				#rx_strTimeStamp = datetime.datetime.strptime(rxFiles_timeStamps[i], '%Y-%m-%d_%H-%M-%S')

				#rx_strTimeStamp = str(rxFiles_timeStamps[i])[-15:]# +'.0'


				#genSyncPlot(axs, frame, startTime, rxFilesInVideo[i], rx_strTimeStamp, **kwargs)
				#if i+1 == len(rxFilesInVideo):
				#	break
				#i+=1


	  # Break the loop
		else:
			print("break")
			break

	# Closes all the frames
	cap.release()

	cv2.destroyAllWindows()

def calcSpeed(d0, d1, c0, c1, delta_t):
	''' Function for simple estimation of object speed based on two time samples
	'''
	c0_x = c0[0]
	c1_x = c1[0]
	c0_y = c0[1]
	c1_y = c1[1]

	print(c0_x)
	print(c1_x)
	s_x = 2*d0*np.tan(np.deg2rad(FOV_X/2))*((c1_x-c0_x)/1280)
	v_x = s_x/delta_t

	s_y = 2*d0*np.tan(np.deg2rad(FOV_Y/2))*((c1_y-c0_y)/720)
	v_y = s_y/delta_t

	v_z = (d1-d0)/(delta_t)

	vtot = np.sqrt(v_x*v_x + v_y*v_y + v_z*v_z)

	return vtot

def calcSize(xmin, ymin, xmax, ymax, d):
	''' Function for calculating size of object based on bounding box coords
		and distance to objects. Assumes a perfect bounding box.
		Inputs:
			- xmin: Minimum x pixel coordinate of bounding box
			- xmax: Max x pixel coordinate of bounding box
			- ymin: Minimum y pixel coordinate of bounding box
			- ymax: Max x pixel coordinate of bounding box
			- d: Distance to object (from echosounder)
		Outputs:
			- Length and Height of object
	'''
	## Calculate size in pizels ##
	px_length = xmax-xmin
	px_height = ymax-ymin

	## Calculate salmon length based on bounding box size, distance and camera FOV ##
	length = 2*d*np.tan((np.deg2rad(FOV_X/2)))*(px_length/1280)
	height = 2*d*np.tan((np.deg2rad(FOV_Y/2)))*(px_height/720)

	length = round(length, 3)
	height = round(height, 3)

	return length, height



def extractTrackedTargets(trackerFile, videoFile):
	''' Reads DeepSORT tracker output file and sorts data by tracked ID and timestamp.
		Input: Path to CSV file with tracker data.
		Output: NA
	'''
	data = []
	timestamps = []
	counter = 0
	print("Current tracker file:", trackerFile)
	print("Current video file:", videoFile)

	with open(trackerFile) as f:
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
		f.close()
	data = np.array((data))

	timestampArr = data[:,0]
	_, idx, inv, count = np.unique(data[:,1], return_index=True, return_inverse=True, return_counts=True)

	ind = np.argsort(inv)
	spp = np.cumsum(count[:-1])
	trackedArray = np.split(data[ind, :], spp, axis=0)

	## Make plot for acoustic data ##
	fig, ax = plt.subplots(1, figsize=(10,6))


	for ID in trackedArray:

		if int(ID[0][1]) not in fusionList_ID:
			continue

		ax.clear()

		currID = ID[0][1]
		print("Current ID:", currID)
		print("Number of tracked timestamps:", len(ID))

		if len(ID) < 6:
			continue
		## Sorting by timestamp to plot in sequence
		data_sorted = sorted(ID, key=lambda x: x[0])

		## Open RX files that matches


		## To exclude tracked objects far away from center ##
		x_threshold_min = 0
		y_threshold_min = 90
		x_threshold_max = 1280-x_threshold_min*2
		y_threshold_max = 720-y_threshold_min*2

		## Initiating array for speed estimate data ##
		temp_speedEstArr = []
		counter = 0

		for sample in data_sorted:
			## Acoustic timestamp ##
			timeStamp = sample[0]
			print("RX timestamp:", timeStamp)
			print("Current sample", sample)

			## Fetching bounding box coordinates for current individual ##
			xmin, ymin, xmax, ymax = int(sample[2]), int(sample[3]), int(sample[4]), int(sample[5])
			print("xmin, ymin, xmax, ymax", xmin, ymin, xmax, ymax)
			if (xmin<x_threshold_min or ymin<y_threshold_min) or (xmax>x_threshold_max or ymax>y_threshold_max):
				print("Individual excluded, too far from center. Continue to next.")
				continue

			## Calculate center of bounding box ##
			x_center = xmin + (xmax-xmin)/2
			y_center = ymin + (ymax-ymin)/2
			center = (int(x_center),int(y_center))
			#cv2.circle(frame, center, 5, color, 5)


			## Extracting frame which matches the timestamp ##
			print("video:",videoFile)
			print("time:",timeStamp)
			frame, vid_timeStamp = getFrame(videoFile, timeStamp)

			## Visualizing bounding box of current tracked individual in focus ##
			color = (0,0,255)
			cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
			#cv2.rectangle(frame, (xmin, ymin-30), (xmin+(len('Salmon')+len(currID))*17, ymin), color, -1)
			cv2.putText(frame, 'Salmon' + "-" + currID,(xmin, ymin-10),0, 0.75, (255,255,255),2)


			## Visualizing exclusion zone to ignore object too high/low in the frame ##
			cv2.rectangle(frame, (x_threshold_min, y_threshold_min, x_threshold_max, y_threshold_max), (0,0,0), 2)
			cv2.namedWindow(str(vid_timeStamp),cv2.WINDOW_NORMAL)
			cv2.resizeWindow(str(vid_timeStamp), 300,300)


			## Fetch acoustic data ##
			echoEnvelope, peaks, rangeVec = getEchoData(timeStamp)

			## Find largest peak ##
			maxPeak = np.max(echoEnvelope[peaks])
			maxPeak_idx = np.argmax(echoEnvelope[peaks])
			## Extract distance to peak ##
			dist = rangeVec[peaks][maxPeak_idx]

			## Get estimate of size based on distance and bounbind box coords ##
			length, height = calcSize(xmin, ymin, xmax, ymax, dist)


			temp_speedEstArr.append([dist, center, timeStamp])

			if len(temp_speedEstArr) >= 2:
				t0 = datetime.datetime.strptime(temp_speedEstArr[-2][2], '%H:%M:%S.%f')
				t1 = datetime.datetime.strptime(temp_speedEstArr[-1][2], '%H:%M:%S.%f')
				delta_t = (t1-t0).total_seconds()
				d0 = temp_speedEstArr[-2][0]
				d1 = temp_speedEstArr[-1][0]
				c0 = temp_speedEstArr[-2][1]
				c1 = temp_speedEstArr[-1][1]

				print(d0, d1, c0, c1, delta_t)

				vtot = calcSpeed(d0, d1, c0, c1, delta_t)

				print("\n\r Estimated swimming speed: ", vtot, "m/s")
				print("Estimated salmon length and height: ", str(length)+"m, ",str(height),"m.")
				cv2.line(frame, c0, c1, (0, 255, 0), thickness=2)




				#delta_t =
			print("\n iubigi", len(temp_speedEstArr))


			cv2.putText(frame, 'Size estimate (length, height): ('+str(length)+', '+str(height)+')',(xmin, ymax+10),0, 0.4, (255,255,255),2)
			cv2.imshow(str(vid_timeStamp), frame)


			ax.plot(rangeVec,echoEnvelope, label='ID:'+ID[0][1]+'. Time:'+timeStamp, alpha=0.3)
			#print("peaks:", peaks)
			#print("Echo peaks:", echoEnvelope[peaks])
			#print("Max:", np.max(echoEnvelope[peaks]))

			#fig2,ax2 = plt.subplots(1)

			#ax2.plot(echoEnvelope, color='red')
			#ax2.plot(echoEnvelope[peaks])
			#plt.show()
			#print("rlen", len(rangeVec))
			#print("maxpeak index:",np.argmax(echoEnvelope[peaks]))



			#print("Max peak:", maxPeak)
			#print("Max peak idx:", maxPeak_idx)
			#ax.plot(peaks, echoEnvelope[peaks], "x", alpha=0.5, label=timeStamp)

			ax.plot(rangeVec[peaks][maxPeak_idx], maxPeak, "x", alpha=0.5, label='Peak', color='red')
			plt.xlim([0,5])
			plt.legend()
			plt.draw()
			#plt.pause(0.001)
			plt.waitforbuttonpress()

			if cv2.waitKey(1) & 0xFF == ord('q'):
				quit()



			#cv2.waitKey(0)
			cv2.destroyAllWindows()
		#time.sleep(2)
		#plt.show()



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
		trackerFiles = loadTrackerData(args.startTime, args.stopTime)
		videoFiles = loadVideoFileNames_fusion(args.startTime, args.stopTime)

		### Må kjøre CSV generering på video 12-04-50! ###
		print("trackerFiles:", trackerFiles)
		print("videos:", videoFiles)
		#print(trackerFiles[0])
		#print(videoFiles[1])



		for i, trackerFile in enumerate(trackerFiles):
			if len(trackerFiles) != len(videoFiles):
				i=i+1
			#fig, axs = plt.subplots(1)
			print("\n")
			print("Current file:", trackerFile)
			print("Current video:", videoFiles[i])
			print("\n")
			extractTrackedTargets(trackerFile, videoFiles[i])
			plt.close()
