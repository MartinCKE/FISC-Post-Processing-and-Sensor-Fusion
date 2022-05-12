#!/usr/bin/python
'''
	Script for fusing YOLOv4-DeepSORT and acoustic data to compute individual fish size and swimming speed.

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
from PIL import Image
import faulthandler


from viewSavedData import loadFileNames
from src.fusePlot import parseVideoTime, get_concat_h_resize
from tools.acousticProcessing import gen_mfilt, matchedFilter, TVG, normalizeData, peakDetect, processEcho

## Manual list of ID's that have valid sequences (does NOT consider duplicates, so still many bad sequences)##
fusionList_ID = [19, 132, 2, 114, 115, 42, 145, 195, 123, 144, 98, 107, 162, 4, 113, 51, 108, 130, 153, 138, 65, 93, 91, 113]

## Intrinsic camera parameters ##
FOV_X = np.deg2rad(48)
FOV_Y = np.deg2rad(28.6)

def genTrackerData(startTime, endTime):
	''' Function for generating CVS-files with DeepSORT tracker information.
		The function will call the external YOLOv4-DeepSORT repository with acoustic ping
		timestamps in the call, such that the DeepSORT tracker information is extracted at
		times where an acoustic sample is acquired.
		Requires GPU computation, recommended to run on CUDA-accelerated PC.
	'''
	rx_files = loadFileNames(startTime, endTime, True, True)
	videoFiles = loadVideoFileNames_fusion(startTime, endTime)

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

		rxFiles_timeStamps =str(rxFiles_timeStamps).strip('[]')

		#DeepSORT_cmd = 'object_tracker.py --weights ./checkpoints/FISC_4 --video '+video+ ' --info --SF_timestamps ' + rxFiles_timeStamps
		#test = '--weights ./checkpoints/FISC_4 --video '+video+ ' --info --SF_timestamps ' + rxFiles_timeStamps
		## Call YOLOv4-DeepSORT script to generate tracker files ###
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
				print("Video added:", filename)


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
				data=np.load(filepath, allow_pickle=True)
				acqInfo = data['header']
				fc = int(acqInfo[0])
				BW = int(acqInfo[1])
				pulseLength = float(acqInfo[2])
				fs = int(acqInfo[3])
				acqInfo = data['header']
				c = 1472.5 ## From profiler data ##

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
				#print("fc:", int(fc), "BW:", int(BW), "fs:", int(fs), \
				#	"plen (us):", int(pulseLength*1e6), "range:", Range, "c:", c, "Downsample step:", downSampleStep)

				## Process echo data and fetch matched filter envelope and peaks in signal ##
				min_idx = int(np.argwhere(rangeVec>1)[0]) ## To ignore detections closer than the set min range
				echoEnvelope, peaks = processEcho(echoData, fc, BW, pulseLength, fs, 1, samplesPerPulse, min_idx)


				return echoEnvelope, peaks, rangeVec


def getFrame(videoFile, timeStamp):
	''' Function for fetching video frame with specific timestamp.
	'''


	startTime, endTime = parseVideoTime(videoFile)
	print("Video Start time and end time:", startTime, endTime)
	timeStamp = datetime.datetime.strptime(timeStamp, '%H:%M:%S.%f')

	### Create a VideoCapture object ###
	cap = cv2.VideoCapture(videoFile)

	### Check if file opened successfully ###
	if (cap.isOpened() == False):
	  print("Unable to read camera feed")

	### Frame resolution is obtained ###
	frame_width = int(cap.get(3))
	frame_height = int(cap.get(4))


	### Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file. ##
	#out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

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

			if abs(timeStamp - last_frame_timestamp) < datetime.timedelta(milliseconds=10) or (timeStamp - last_frame_timestamp).days == -1:
				### Fetches a frame with timestamp coinciding with acoustic ping timestamp (10 ms accuracy) ##
				#print("rx file:", timeStamp)
				print("\r\nMATCH")
				#print("Video CURRENT timestamp:", curr_frame_timestamp)
				return frame, last_frame_timestamp

	  # Break the loop
		else:
			print("Could not read video. Check CV2 build.")
			break

	# Closes all the frames
	cap.release()

	cv2.destroyAllWindows()

def calcSpeed_old(d0, d1, c0, c1, delta_t):
	''' Function for simple estimation of object speed based on two time samples.
		Discarded due to wrong geometrical assumptions, but not deleted for future
		reference.
	'''
	c0_x = c0[0]
	c1_x = c1[0]
	c0_y = c0[1]
	c1_y = c1[1]

	s_x = 2*d0*np.tan(FOV_X/2)*((c1_x-c0_x)/1280)
	v_x = s_x/delta_t

	s_y = 2*d0*np.tan(FOV_Y/2)*((c1_y-c0_y)/720)
	v_y = s_y/delta_t

	v_z = (d1-d0)/(delta_t)

	vtot = np.sqrt(v_x*v_x + v_y*v_y + v_z*v_z)

	return vtot

def calcSize_old(xmin, ymin, xmax, ymax, d):
	''' Function for calculating size of object based on bounding box coords
		and distance to objects. Assumes a perfect bounding box.
		Not used due to wrong geometrical assumptions, but not deleted for future
		reference.
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
	length = 2*d*np.tan((FOV_X/2))*(px_length/1280)
	height = 2*d*np.tan((FOV_Y/2))*(px_height/720)

	length = round(length, 3)
	height = round(height, 3)

	return length, height

def calcCoords(d, c):
	''' Function for calculating relative 3D Cartesian coordinates (in m) for
		detected objects based on echosounder distance and bounding box center
		coordinates.
		Inputs:
			-d: Echosounder distance to object
			-c: Pixel coordinate of bounding box center
		Outputs:
			[X, Y, Z]: Cartesian coordinates
	'''

	c_x = c[0]
	c_y = c[1]
	#d = 50.3
	#c_x = 426
	#c_y = 322

	## Tape-test, seems logical ##
	#d = 53.5
	#c_x = 272
	#c_y = 100

	phi_x = np.arctan( (((1280/2)-c_x) *np.tan(FOV_X/2)) / (1280/2))
	phi_y = np.arctan( (((720/2)-c_y) *np.tan(FOV_Y/2)) / (720/2))

	x_coord = d * np.sin(phi_x)
	y_coord = d * np.sin(phi_y)
	z_coord = np.sqrt(d**2 - x_coord**2 - y_coord**2)


	return x_coord, y_coord, z_coord

def calcSize(xmin, ymin, xmax, ymax, z):
	''' Estimate size of object in frame based on distance to object and bounding box size.
		Inputs:
			-xmin: Minimum x coordinate of bounding box
			-ymin: Minimum y coordinate of bounding box
			-xmax: Maximum x coordinate of bounding box
			-ymax: Maximum y coordinate of bounding box
			-d: True distance to object
		Outputs:
			[length, height] in meters
	'''
	## Calculate size in pizels ##
	px_length = xmax-xmin
	px_height = ymax-ymin

	## Estimate salmon length based on bounding box size, distance and camera FOV ##
	length = 2*z*np.tan((FOV_X/2))*(px_length/1280)
	height = 2*z*np.tan((FOV_Y/2))*(px_height/720)

	length = round(length, 3)
	height = round(height, 3)

	return length, height

def calcSpeed(d_t0, d_t1):
	''' Function for esimating swimming speed based on coordinate change
		between two frames
		Inputs:
			- d_t0: [x, y, z, timestamp] at time t1
			- d_t1: [x, y, z, timestamp] at time t1
		Output:
			- Estimated swimming speed (magnitude)
	'''

	t0 = datetime.datetime.strptime(d_t0[-1], '%H:%M:%S.%f')
	t1 = datetime.datetime.strptime(d_t1[-1], '%H:%M:%S.%f')
	delta_t = (t1-t0).total_seconds()

	delta_pos = np.subtract(d_t1[0:-1], d_t0[0:-1])
	speed = delta_pos/delta_t
	print("Velocities in [x, y, z] directions:", speed)
	tot_speed = np.linalg.norm(speed)
	print("Absolute velocity", tot_speed, "m/s")
	return tot_speed

def extractTrackedTargets(trackerFile, videoFile, savePlots, showID):
	''' Reads DeepSORT tracker CSV file and visualizes time-synchronized plots
		of individually tracked salmon.
		Matches frame with coinciding acoustic ping for sensor fusion.
		Input: Path to CSV file with tracker data and matching video file.
		Arguments:
			- savePlots: Export synchronized plots
			- showID: Only show matches with salmon ID#
	'''
	data = []
	timestamps = []
	counter = 0
	#print("Current tracker file:", trackerFile)
	#print("Current video file:", videoFile)

	with open(trackerFile) as f:
		data_reader = csv.reader(f)
		for i, line in enumerate(data_reader):
			if i == 0:
				print("header:", line)
				continue
			ts = line[0][3:-2]

			ID = int(line[1])
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

	if showID == 0:
		fig, acousticAX = plt.subplots(1)#, figsize=(10,6))

	for track in trackedArray:
		currID = int(track[0][1])

		if currID not in fusionList_ID:
			print("Chosen ID "+'"'+str(currID)+'"'+" is not in list of good fusion matches. Please use one of the following ID's:\r")
			print("[19, 132, 2, 114, 115, 42, 145, 195, 123, 144, 98, 107, 162, 4, 113, 51, 108, 130, 153, 138, 65, 93, 91, 113]\n\r")
			continue
		if showID != 0:
			if currID != showID:
				continue
			else:
				fig, acousticAX = plt.subplots(1)#, figsize=(10,6))

		acousticAX.clear()

		print("Current ID:", currID)
		print("Number of tracked timestamps:", len(track))

		if len(track) < 6:
			continue

		## Sorting by timestamp to plot in sequence ##
		data_sorted = sorted(track, key=lambda x: x[0])

		## To exclude tracked objects far away from center ##
		x_threshold_min = 0
		y_threshold_min = 90
		x_threshold_max = 1280-x_threshold_min*2
		y_threshold_max = 720-y_threshold_min*2

		## Initiating array for speed estimate data ##
		temp_speedEstArr = []
		temp_speedEstArr_old = []
		centerArr = []
		counter = 0

		for sample in data_sorted:
			## Acoustic timestamp ##
			timeStamp = sample[0]

			## Fetching bounding box coordinates for current individual ##
			xmin, ymin, xmax, ymax = int(sample[2]), int(sample[3]), int(sample[4]), int(sample[5])
			#print("xmin, ymin, xmax, ymax", xmin, ymin, xmax, ymax)
			if (xmin<x_threshold_min or ymin<y_threshold_min) or (xmax>x_threshold_max or ymax>y_threshold_max):
				print("Individual excluded, too far from center. Continue to next.")
				continue

			## Calculate center of bounding box ##
			x_center = xmin + (xmax-xmin)/2
			y_center = ymin + (ymax-ymin)/2
			center = (int(x_center),int(y_center))

			centerArr.append(center)


			## Extracting frame which matches the timestamp ##
			frame, vid_timeStamp = getFrame(videoFile, timeStamp)


			## Visualizing bounding box of current tracked individual in focus ##
			color = (0,0,255)

			cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2) ## Bounding Box
			cv2.putText(frame, 'Salmon' + "-" + str(currID),(xmin, ymin-10),0, 0.75, (255,255,255),2)


			## Visualizing exclusion zone to ignore object too high/low in the frame ##
			cv2.rectangle(frame, (x_threshold_min, y_threshold_min, x_threshold_max, y_threshold_max), (0,0,0), 2)
			cv2.namedWindow(str(vid_timeStamp),cv2.WINDOW_NORMAL)
			cv2.resizeWindow(str(vid_timeStamp), 300,300)


			## Fetch acoustic data ##
			echoEnvelope, peaks, rangeVec = getEchoData(timeStamp)

			## Find largest peak ##
			maxPeak = np.max(echoEnvelope[peaks])

			#temp_Envelope = echoEnvelope ## To ignore detections very close
			#temp_Envelope[0:min_idx] = 0
			#maxPeak_idx = np.argmax(temp_Envelope[peaks])
			maxPeak_idx = np.argmax(echoEnvelope[peaks])
			#maxPeak_idx = np.argmax(np.argwhere(echoEnvelope[peaks]>1))
			## Extract distance to peak ##
			dist = rangeVec[peaks][maxPeak_idx]


			x, y, z = calcCoords(dist, center)
			print("Coordinates for object [x, y, z] (m):", x, y, z)
			print("Echosounder distance (m):", dist)

			## Get estimate of size based on distance and bounbind box coords ##
			length_old, height_old = calcSize_old(xmin, ymin, xmax, ymax, dist)
			length, height = calcSize(xmin, ymin, xmax, ymax, z)

			## Preview size estimate on frame ##
			sizeText = 'Size estimate (length, height): ('+str(length)+', '+str(height)+') [m]'
			(w, h), _ = cv2.getTextSize(sizeText, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) ## Space required by text

			#cv2.rectangle(frame, (xmin, ymax + 40), (xmin + w, ymax), (0,0,0), -1) ## Background for text
			cv2.rectangle(frame, (xmin-3, ymax+3), (xmin + w+12, ymax+65), (0,0,0), -1) ## Background for text
			cv2.putText(frame, sizeText,(xmin, ymax+25),0, 0.7, (255,255,255),2, cv2.LINE_AA) ## Write text

			temp_speedEstArr.append([x, y, z, timeStamp])
			temp_speedEstArr_old.append([dist, center, timeStamp])

			color = next(acousticAX._get_lines.prop_cycler)['color']
			acousticAX.plot(rangeVec,echoEnvelope, label='ID:'+str(currID)+'. Time:'+timeStamp, alpha=0.3, color=color)
			acousticAX.set_xlabel("Range [m]")
			acousticAX.set_ylabel("Matched Filter Output")
			acousticAX.set_title("Acoustic Data for Fish #"+str(currID))#ID[0][1])

			acousticAX.plot(rangeVec[peaks][maxPeak_idx], maxPeak, "x", alpha=0.5, label='Peak at '+str(rangeVec[peaks][maxPeak_idx])[0:4]+'m used.', color=color)

			plt.xlim([0,5])
			plt.draw()
			plt.pause(0.001)

			if len(temp_speedEstArr) >= 2:

				while len(acousticAX.lines) > 4:
					## To only display 2 last acoustic pings which are used in speed esimates ##
					acousticAX.lines.pop(0)

				plt.legend()
				plt.draw()

				t0 = datetime.datetime.strptime(temp_speedEstArr_old[-2][-1], '%H:%M:%S.%f')
				t1 = datetime.datetime.strptime(temp_speedEstArr_old[-1][-1], '%H:%M:%S.%f')
				delta_t = (t1-t0).total_seconds()
				d0 = temp_speedEstArr_old[-2][0]
				d1 = temp_speedEstArr_old[-1][0]
				c0 = centerArr[-2]
				c1 = centerArr[-1]

				speed = calcSpeed(temp_speedEstArr[-2], temp_speedEstArr[-1])

				#print("Estimated salmon length and height: ", str(length)+"m, ",str(height),"m.")
				cv2.line(frame, c0, c1, (0, 255, 0), thickness=2)
				cv2.circle(frame, c1, 1, (0, 255, 0), 10) # Show green circle in current frame bounding box center
				cv2.circle(frame, c0, 1, (0, 255, 255), 10) # Show yellow circle in previous frame bounding box center
				BLspeed = speed/length

				speedText = 'Swimming speed estimate: ' + str(speed)[0:4]+' m/s or '+str(BLspeed)[0:4]+' BL/s'
				(w_speed, h_speed), _ = cv2.getTextSize(speedText, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) ## Space required by text
				cv2.putText(frame, speedText, (xmin, ymax+60),0, 0.7, (255,255,255),2, cv2.LINE_AA) ## Write text


				if savePlots:
					video_timeStamp = str(vid_timeStamp)[11:-1].replace(":", ".")
					rxFile_timeStamp = timeStamp.replace(":", ".")
					frame_filename = 'VideoFrame_2022-02-16_'+video_timeStamp+'.png'
					RX_filename = 'RXSample_2022-02-16_'+rxFile_timeStamp+'.png'
					syncPlotName = 'syncPlot_DeepSORT_2022-02-16_'+rxFile_timeStamp+'.png'

					cv2.imwrite('temp/'+frame_filename, frame)
					plt.savefig('temp/'+RX_filename, dpi=600)

					im1 = Image.open('temp/'+frame_filename)
					im2 = Image.open('temp/'+RX_filename)

					syncPlotName = 'syncPlot_DeepSORT_sectorFocus_2022-03-11_'+rxFile_timeStamp+'.png'

					get_concat_h_resize(im1, im2).save('generatedPlots/'+syncPlotName)
					print(os.getcwd())

					os.remove(os.getcwd()+'/temp/'+frame_filename)
					os.remove(os.getcwd()+'/temp/'+RX_filename)


			cv2.imshow(str(vid_timeStamp), frame) ## Display video frame

			if not savePlots:
				plt.waitforbuttonpress()


			if (cv2.waitKey(1) & 0xFF) == ord('q'):
				quit()


			cv2.destroyAllWindows()



def loadTrackerData(startTime, endTime):
	''' Function for loading YOLOv4-DeepSORT CSV tracker file paths.
		Inputs:
			- [startTime, endTime]: Start and end time to load files within.
	'''
	directory = os.getcwd()+'/deepsort'
	trackerFiles = []
	for root, dirs, filenames in os.walk(directory, topdown=False):
		for filename in filenames:
			if 'DS' in filename or not filename.endswith('.csv'):
				continue
			hhmm = str(re.findall('[0-9]{2}-[0-9]{2}-[0-9]{2}', filename))[14:-5]
			hhmm = hhmm.replace("-", ":")
			if startTime <= hhmm <= endTime:
				trackerFiles.append(root+'/'+filename)

	trackerFiles = sort(trackerFiles) ## Sorting by time

	return trackerFiles


if __name__ == '__main__':
	faulthandler.disable()
	faulthandler.enable(all_threads=True)

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
	parser.add_argument(
		"--savePlots",
		action="store_true",
		dest="savePlots",
		help="Export time-synchronized video frame and acoustic samples.",
	)
	parser.add_argument(
		"--ID",
		action="store",
		type=int,
		default=0,
		dest="showID",
		help="To only process fusion data for ID #N.",
	)

	args = parser.parse_args()

	if args.generate:
		genTrackerData(args.startTime, args.stopTime)
	if args.process:
		trackerFiles = loadTrackerData(args.startTime, args.stopTime)
		videoFiles = loadVideoFileNames_fusion(args.startTime, args.stopTime)

		print("trackerFiles:", trackerFiles)
		print("videos:", videoFiles)


		for i, trackerFile in enumerate(trackerFiles):
			## Running sensor fusion on every video file iteratively ##
			if len(trackerFiles) != len(videoFiles):
				i=i+1
			print("\n")
			print("Current file:", trackerFile)
			print("Current video:", videoFiles[i])
			print("\n")
			extractTrackedTargets(trackerFile, videoFiles[i], args.savePlots, args.showID)
			plt.close()
