import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.signal
from pylab import *
import time
import re
from datetime import datetime
import argparse

#Importing other scripts
from src.fusePlot import syncPlot_timeStampFromFrames
from tools.acousticProcessing import matchedFilter, gen_mfilt, peakDetect, processEcho
from src.profilePlot import SVProfilePlot, profilePlot
from src.IMU import plotIMUData, inclination_current


matplotlib.use('TkAgg')
channelArray = [['Sector 1', '0'],['Sector 2', '2'],
				['Sector 3', '2'],['Sector 4', '0'],
				['Sector 5', '1'],['Sector 6', '3'],
				['Sector 7', '3'],['Sector 8', '1']]


## Change this to look at certain dates/hours/minutes
## SavedData-folder must be in same directory as script
#directory = os.getcwd()+'/Data/SectorFocus/16-02-22'#09_16'
#print("Directory:", directory)



headingDict = {"N":0, "NNE":22.5, "NE":45, "ENE":67.5, "E":90,\
			   "ESE":112.5, "SE":135, "SSE":157.5, "S":180, \
			   "SSW":202.5, "SW":225, "WSW":247.5, "W":270, \
			   "WNW":292.5, "NW":315, "NNW":337.5}


def normalizeData(data):
	if not np.all((data == 0)):
		return (data - np.min(data)) / (np.max(data) - np.min(data))
	else:
		print('Only zeros ecountered, check data.')
		return data

def peakDetect(data, num_train=6, num_guard=2, rate_fa=1e-3):
	''' Cell-Averaging Constant False Alarm Rate (CFAR) detector algorithm.

		CUT = Cell under test
		Parameters:
		num_train = samples surrounding CUT, assumed to be noise
		num_guard = samples adjacent to CUT to avoid signal leakage to noise
		rate_fa = chosen false alarm rate (default = 0.001)
	'''

	num_cells = data.size
	num_train_half = round(num_train / 2)
	num_guard_half = round(num_guard / 2)
	num_side = num_train_half + num_guard_half

	alpha = num_train*(rate_fa**(-1/num_train) - 1) # threshold factor

	peak_idx = []
	peaks = []
	noiseArr = []
	thresholdArr = []

	for i in range(0, num_cells):

		if i<num_side:
			'''Portion of samples before CUT is num_side samples out '''
			trainingCellSum = np.sum(data[:i+num_side+1])*2
			guardCellSum = np.sum(data[:i+num_guard_half+1])*2
			p_noise = (trainingCellSum - guardCellSum) / num_train

		elif i>num_cells-num_side:
			'''Portion of samples after CUT is num_side samples from end '''
			trainingCellSum = np.sum(data[i-num_side:])*2
			guardCellSum = np.sum(data[i-num_guard_half:])*2
			p_noise = (trainingCellSum - guardCellSum) / num_train

		else:
			trainingCellSum = np.sum(data[i-num_side:i+num_side+1])
			guardCellSum = np.sum(data[i-num_guard_half:i+num_guard_half+1])
			p_noise = (trainingCellSum - guardCellSum) / num_train

		#alpha = 2 ## Custom threshold (rate_fa ignored)
		threshold = alpha * p_noise
		thresholdArr.append(threshold)
		noiseArr.append(p_noise)

		if data[i] > threshold:
			peak_idx.append(i)
			peaks.append(p_noise)
			if peaks[i] <= 0:
				peaks[i] = 0.00001

		else:
			peaks.append(data[i])
			if peaks[i] <= 0:
				peaks[i] = 0.00001


	peak_idx = np.array(peak_idx, dtype=int)

	detectorarr = np.log10(data/peaks)

	return peak_idx, noiseArr, detectorarr, thresholdArr


def move_figure(f, x, y):
	"""Move figure's upper left corner to pixel (x, y)"""
	backend = matplotlib.get_backend()
	if backend == 'TkAgg':
		f.canvas.manager.window.wm_geometry("+%d+%d" % (x, y))
	elif backend == 'WXAgg':
		f.canvas.manager.window.SetPosition((x, y))
	else:
		# This works for QT and GTK
		# You can also use window.setGeometry
		f.canvas.manager.window.move(x, y)

def loadFileNames(startTime, stopTime, sectorFocus, ace):

	if ace:
		if sectorFocus:
			directory = os.getcwd()+'/data/SectorFocus/11-03-22'
		else:
			directory = os.getcwd()+'/data/11-03-22'
	else:
		if sectorFocus:
			directory = os.getcwd()+'/data/SectorFocus/16-02-22'
		else:
			directory = os.getcwd()+'/data/16-02-22'

	files = []
	hhmmss_list = []

	#hhmmss = str(re.findall('[0-9]{2}:[0-9]{2}:[0-9]{2}', filename))[2:-2]

	for root, dirs, filenames in os.walk(directory, topdown=False):
		for filename in filenames:
			if '' in filename:
				filename = filename.replace("", ":")
			hhmm = str(re.findall('[0-9]{2}:[0-9]{2}', filename))[2:-2] ## To get HH:MM from filename
			#print(hhmm)
			#print(directory)
			#print(filename)
			if 'DS' in filename:
				continue

			## Only add desired files to list
			if startTime <= hhmm <= stopTime and filename.endswith('.npz'):
				files.append(root+'/'+filename)
				print("added file:", filename)
				#hhmmss_list.append(hhmmss)

	## Sorting files by time
	files = sorted(files, key=lambda x: x[-18:-4])

	return files

def loadVideoFileNames(startTime, stopTime, ace, deepsort):
	videofiles = []
	hhmmss_list = []
	if ace and not deepsort:
		directory = os.getcwd()+'/Data/cam_recordings/secondTest/compressed'
		print("Ace")
	elif ace and deepsort:
		directory = os.getcwd() + '/deepsort/outputs'
		print("HEU")
	else:
		print("nope")
		directory = os.getcwd()+'/Data/cam_recordings/firstTest'

	#hhmm = str(re.findall('[0-9]{2}-[0-9]{2}', filename))[2:-2]
	#hhmmss = str(re.findall('[0-9]{2}-[0-9]{2}-[0-9]{2}', filename))[2:-2]
	print("Dir:", directory)
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
		default='00:00',
		dest='startTime',
		help="View data acquired from NN:NN o'clock",
	)
	parser.add_argument(
		"--stop",
		action="store",
		type=str,
		default='23:59',
		dest="stopTime",
		help="View data acquired until NN:NN o'clock",
	)
	parser.add_argument(
		"--sf",
		action="store_true",
		dest="sectorFocus",
		help="Process sector focus data only",
	)
	parser.add_argument(
		"--syncPlot",
		action="store_true",
		dest="syncPlot",
		help="Plot synchronized video-frame and RX data",
	)
	parser.add_argument(
		"--savePlots",
		action="store_true",
		dest="savePlots",
		help="Export synchronized plots to single images",
	)
	parser.add_argument(
		"--show",
		action="store_true",
		dest="showPlots",
		help="Show plots or not",
	)
	parser.add_argument(
		"--o2temp",
		action="store_true",
		dest="o2temp",
		help="Parse and plot O2 + Temp data",
	)
	parser.add_argument(
		"--profile",
		action="store_true",
		dest="profile",
		help="Parse and plot SV, O2 + Temp data as function of depth",
	)
	parser.add_argument(
		"--imu",
		action="store_true",
		dest="imu",
		help="Parse and plot IMU data over time",
	)
	parser.add_argument(
		"--ace",
		action="store_true",
		dest="ace",
		help="Add this for 2nd field test data (Sintef Ace, Frøya)",
	)
	parser.add_argument(
		"--deepsort",
		action="store_true",
		dest="deepsort",
		help="Fetch videos which have gone through DeepSORT tracking.",
	)
	args = parser.parse_args()




	if args.syncPlot and not args.o2temp:
		rx_files = loadFileNames(args.startTime, args.stopTime, args.sectorFocus, args.ace)
		videoFiles = loadVideoFileNames(args.startTime, args.stopTime, args.ace, args.deepsort)
		print("RX files:", rx_files)
		print("Videos:", videoFiles)
		for video in videoFiles:
			syncPlot_timeStampFromFrames(video, rx_files, sectorFocus=args.sectorFocus, \
												savePlots=args.savePlots, showPlots=args.showPlots, ace=args.ace, deepsort=args.deepsort)

		quit()
	if args.o2temp:
		if args.profile:
			SVProfilePlot(args.ace)
			## Depth measurements performed in this time window
			if args.ace:
				files = loadFileNames('12:50', '13:00', False, args.ace)
			else:
				files = loadFileNames('09:47', '09:51', args.sectorFocus, args.ace)
			#for file in files:
			#    print(file)
			profilePlot(files, args.ace)
		else:
			files = loadFileNames(args.startTime, args.stopTime, args.sectorFocus)
			for file in files:
				print(file)#print(files)
			O2TempPlot(files)

	if args.imu:
		if args.ace:
			files = loadFileNames('10:58', args.stopTime, args.sectorFocus, args.ace)
		else:
			files = loadFileNames('10:00', args.stopTime, args.sectorFocus, args.ace)
		plotIMUData(files, args.ace)




	#fig, ax = plt.subplots(1)
	rx_files = loadFileNames(args.startTime, args.stopTime, args.sectorFocus, args.ace)

	#fig3, (ax1, ax2) = plt.subplots(2,figsize=(8,6))

	for filename in rx_files:
		timeStamp = str(filename)[-18:-4]
		date = str(filename)[-45:-37]

		data=np.load(filename, allow_pickle=True)

		print("Current file:", filename[-46:-4])


		if len(data['header']) > 5:
			''' Since header-contents were changed at some point '''
			acqInfo = data['header']
			imuData = data['IMU']
			O2Data = data['O2']
			print("O2 Data", O2Data)
			fc = int(acqInfo[0])
			BW = int(acqInfo[1])
			pulseLength = acqInfo[2]
			fs = acqInfo[3]
			Range = int(acqInfo[4])
			c = acqInfo[5]
			downSampleStep = int(acqInfo[6])

		else:
			acqInfo = data['header']
			imuData = data['IMU']
			print(acqInfo)
			fs = acqInfo[0]
			Range = acqInfo[1]
			pulseLength = acqInfo[2]
			c = acqInfo[3]
			downSampleStep = int(acqInfo[4])

		print("fc:", int(fc), "BW:", int(BW), "fs:", int(fs), \
			"plen (us):", int(pulseLength*1e6), "range:", Range, "c:", c, "Downsample step:", downSampleStep)


		if args.ace:
			c = 1472.5
		else:
			c = 1463
		#downSampleStep = 1

		## Fetching O2 and Temp value. Sensor data structure is odd, length varies ##
		tempArray = np.array_str(O2Data).strip(' []\n')
		dataArr = tempArray.split(" ")
		print("\n\r hei:", len(dataArr))
		print("O2 data:", dataArr)
		if len(dataArr) == 6:
			O2 = float(dataArr[2])
		elif len(dataArr) == 8:
			try:
				O2 = float(dataArr[4])
			except:
				O2 = float(dataArr[3])
		elif len(dataArr) == 7:
			try:
				O2 = float(dataArr[3])
			except:
				O2 = float(dataArr[2])
		else:
			O2 = float(dataArr[2])
		Temp = float(dataArr[-1])

		## Differentiate sector scan from sector focus
		if data['sectorData'].ndim == 1:
			Sector4_data = data['sectorData'][:]
			nSamples = len(Sector4_data)
			sectorFocus = True
			print("Sector Focus")
		else:
			Sector1_data = data['sectorData'][:,0]
			Sector2_data = data['sectorData'][:,1]
			Sector3_data = data['sectorData'][:,2]
			Sector4_data = data['sectorData'][:,3]
			Sector5_data = data['sectorData'][:,4]
			Sector6_data = data['sectorData'][:,5]
			Sector7_data = data['sectorData'][:,6]
			Sector8_data = data['sectorData'][:,7]
			nSamples = len(Sector1_data)
			print("Sector Scan")


		### Acquisition constants ###
		#SampleTime = Range*2.0/c # How long should we sample for to cover range
		SampleTime = nSamples*(1/fs)
		Range = c*SampleTime/2
		#nSamples = int(fs*SampleTime) # Number og samples to acquire per ping
		samplesPerPulse = int(fs*pulseLength)  # How many samples do we get per pulse length
		tVec = np.linspace(0, SampleTime, nSamples)
		tVecShort = tVec[0:len(tVec):downSampleStep] # Downsampled time vector for plotting
		plen_d = (c*pulseLength)/2
		rangeVec = np.linspace(-plen_d, Range, len(tVec))
		rangeVecShort = np.linspace(-plen_d, Range, len(tVecShort)).round(decimals=2)


		## Matched filter
		mfilt = gen_mfilt(fc, BW, pulseLength, fs)


		polarPlot_init(tVecShort, rangeVecShort)

		#plt.xlim((0, Range))
		#fig3, (ax1, ax2) = plt.subplots(2,figsize=(7,6))

		if args.sectorFocus:
			#fig2, ax = plt.subplots(1,figsize=(7,6))

			#move_figure(fig2, 600, 0)
			roll = imuData[0]
			pitch = imuData[1]
			heading = imuData[2]

			a = np.arctan(np.deg2rad(roll))*np.arctan(np.deg2rad(roll))
			b = np.arctan(np.deg2rad(pitch))*np.arctan(np.deg2rad(pitch))
			tilt = np.degrees(np.arctan(sqrt(a+b)))

			inclination, currentDir = inclination_current(roll, pitch, heading)
			heading = round(heading, 2)

			min_idx = int(np.argwhere(rangeVecShort>1)[0]) ## To ignore detections closer than the set min range
			echoEnvelope, peaks = processEcho(Sector4_data, fc, BW, pulseLength, fs, downSampleStep, samplesPerPulse, min_idx)


			#CH1_Env, _ = matchedFilter(Sector4_data, Sector4_data, mfilt, downSampleStep, samplesPerPulse)
			CH1_peaks_idx, CH1_noise, CH1_detections, CH1_thresholdArr = peakDetect(echoEnvelope, num_train=3, num_guard=5, rate_fa=0.3)
			CH1_Intensity, _ = colorMapping(echoEnvelope, echoEnvelope)

			detectionArr = np.zeros((len(echoEnvelope)))

			detectionArr[peaks] = echoEnvelope[peaks]

			maxPeak = np.max(echoEnvelope[peaks])

			maxPeak_idx = np.argmax(echoEnvelope[peaks])
			## Extract distance to peak ##
			dist = rangeVec[peaks][maxPeak_idx]
			#for val in CH1_Intensity:
			#	print(val)
			#quit()

			RX_polarPlot(CH1_Intensity, _, 2, [0,0], detectionArr, inclination, currentDir, heading, O2, Temp, filename, sectorFocus=True)
			plt.savefig(os.getcwd()+'/plots/PolarPlot_'+timeStamp+'.pdf')
			'''
			#plt.show()
			#continue
			ax.clear()
			#ax1.clear()

			ax.plot(rangeVecShort,echoEnvelope,color='red')
			ax.plot(rangeVecShort[peaks], echoEnvelope[peaks], 'x', color='black', alpha=0.5, label='CA-CFAR Detections')
			ax.set_xlabel("Range [m]")
			ax.set_ylabel("Matched Filter Output")
			ax.set_title("Sector 4 Acoustic Data")#Acoustic Data for Ping "+str(date)+'_'+timeStamp)# for Fish #"+ID[0][1])
			ax.plot(rangeVecShort[peaks][maxPeak_idx], maxPeak, "x", alpha=1, color='blue', label='Peak at '+str(rangeVecShort[peaks][maxPeak_idx])[0:4]+'m used.')

			ax.set_xlim([0,5])
			#ax2.set_xlim([0,5])
			fig2.suptitle("Acoustic Data for Ping "+str(date)+'_'+timeStamp)



			## Acoustic processing pipeline plot ##
			ax1.plot(rangeVec, Sector4_data, label='Raw RX Data')
			ax1.set_xlabel("Range [m]")
			ax1.set_ylabel("Signal Amplitude [V]")
			ax1.set_title('Raw RX Data')

			ax2.plot(rangeVecShort,echoEnvelope,color='red')
			ax2.plot(rangeVecShort[peaks], echoEnvelope[peaks], 'x', color='black', alpha=0.5, label='CA-CFAR Detections')
			ax2.set_xlabel("Range [m]")
			ax2.set_ylabel("Matched Filter Output")
			ax2.set_title("Processed RX Data")#Acoustic Data for Ping "+str(date)+'_'+timeStamp)# for Fish #"+ID[0][1])
			ax2.plot(rangeVecShort[peaks][maxPeak_idx], maxPeak, "x", alpha=1, color='blue', label='Peak at '+str(rangeVecShort[peaks][maxPeak_idx])[0:4]+'m used.')

			ax1.set_xlim([0,5])
			ax2.set_xlim([0,5])
			fig3.suptitle("Acoustic Data for Ping "+str(date)+'_'+timeStamp)

			plt.tight_layout()
			##


			#ax.plot(rangeVecShort, CH1_Env, label='Signal from Sector 4')
			ax.legend()
			ax.legend()
			'''
			plt.draw()
			plt.pause(1e-6)
			plt.waitforbuttonpress()
			#plt.savefig(os.getcwd()+'/plots/latest_SectorFocus.pdf')
			#plt.savefig(os.getcwd()+'/plots/acousticProcessing_example.pdf')
			#plt.show()


			continue

		else:
			#fig2, ax2 = plt.subplots(2,figsize=(7,6))
			inclinationArr = []
			currentDirArr = []
			headingArr = []
			for zone in range(1,5):
				#ax2[0].clear()
				#ax2[1].clear()
				roll = imuData[zone-1][0]
				pitch = imuData[zone-1][1]
				heading = imuData[zone-1][2]

				headingArr.append(round(heading, 2))

				inclination, currentDir = inclination_current(roll, pitch, heading)
				inclinationArr.append(inclination)
				currentDirArr.append(currentDir)
				#print("INCLINATION:", inclination, "currentAngle", currentDir)

				min_idx = int(np.argwhere(rangeVecShort>1)[0]) ## To ignore detections closer than the set min range
				CH1_Data = data['sectorData'][:,2*zone-1]
				CH2_Data = data['sectorData'][:,2*(zone-1)]


				echoEnvelope_CH1, peaks_CH1 = processEcho(CH1_Data, fc, BW, pulseLength, fs, downSampleStep, samplesPerPulse, min_idx)
				echoEnvelope_CH2, peaks_CH2 = processEcho(CH2_Data, fc, BW, pulseLength, fs, downSampleStep, samplesPerPulse, min_idx)

				CH1_peaks_idx, CH1_noise, CH1_detections, CH1_thresholdArr = peakDetect(echoEnvelope_CH1, num_train=3, num_guard=5, rate_fa=0.3)
				CH2_peaks_idx, CH2_noise, CH2_detections, CH2_thresholdArr = peakDetect(echoEnvelope_CH2, num_train=3, num_guard=5, rate_fa=0.3)
				CH1_Intensity, CH2_Intensity = colorMapping(echoEnvelope_CH1, echoEnvelope_CH2)

				detectionArr_CH1 = np.zeros((len(echoEnvelope_CH1)))
				detectionArr_CH2 = np.zeros((len(echoEnvelope_CH2)))

				detectionArr_CH1[peaks_CH1] = echoEnvelope_CH1[peaks_CH1]
				detectionArr_CH2[peaks_CH2] = echoEnvelope_CH2[peaks_CH2]

				maxPeak_CH1 = np.max(echoEnvelope_CH1[peaks_CH1])
				maxPeak_CH2 = np.max(echoEnvelope_CH2[peaks_CH2])

				maxPeak_idx_CH1 = np.argmax(echoEnvelope_CH1[peaks_CH1])
				maxPeak_idx_CH2 = np.argmax(echoEnvelope_CH2[peaks_CH2])
				## Extract distance to peak ##
				dist_CH1 = rangeVec[peaks_CH1][maxPeak_idx_CH1]
				dist_CH2 = rangeVec[peaks_CH2][maxPeak_idx_CH2]



				RX_polarPlot(CH1_Intensity, CH2_Intensity, zone, CH1_detections, CH2_detections, \
							inclinationArr, currentDirArr, headingArr, O2, Temp, filename, sectorFocus=False)
				#plt.savefig(os.getcwd()+'/plots/PolarPlot_'+timeStamp+'.pdf')

				'''
				ax2[0].plot(rangeVecShort,echoEnvelope_CH1,color='red')
				ax2[0].plot(rangeVecShort[peaks_CH1], echoEnvelope_CH1[peaks_CH1], 'x', color='black', alpha=0.5, label='CA-CFAR Detections')
				ax2[0].set_xlabel("Range [m]")
				ax2[0].set_ylabel("Matched Filter Output")
				ax2[0].set_title("Processed RX Data for Sector "+channelArray[2*(zone-1)][0])#Acoustic Data for Ping "+str(date)+'_'+timeStamp)# for Fish #"+ID[0][1])
				ax2[0].plot(rangeVecShort[peaks_CH1][maxPeak_idx_CH1], maxPeak_CH1, "x", alpha=1, color='blue', label='Peak at '+str(rangeVecShort[peaks_CH1][maxPeak_idx_CH1])[0:4]+'m used.')

				ax2[1].plot(rangeVecShort,echoEnvelope_CH2,color='red')
				ax2[1].plot(rangeVecShort[peaks_CH2], echoEnvelope_CH2[peaks_CH2], 'x', color='black', alpha=0.5, label='CA-CFAR Detections')
				ax2[1].set_xlabel("Range [m]")
				ax2[1].set_ylabel("Matched Filter Output")
				ax2[1].set_title("Processed RX Data for Sector "+channelArray[zone*2-1][0])#Acoustic Data for Ping "+str(date)+'_'+timeStamp)# for Fish #"+ID[0][1])
				ax2[1].plot(rangeVecShort[peaks_CH2][maxPeak_idx_CH2], maxPeak_CH2, "x", alpha=1, color='blue', label='Peak at '+str(rangeVecShort[peaks_CH2][maxPeak_idx_CH2])[0:4]+'m used.')

				#ax2[0].clear()
				#ax2[1].clear()
				#plt.subplots(211)
				#ax2[0].plot(rangeVec, CH1_Samples, label='Signal from '+channelArray[2*(zone-1)][0])
				##ax2[0].plot(rangeVecShort, CH1_Env, label='Signal from '+channelArray[2*(zone-1)][0])
				#ax2[0].plot(freqs, CH1_fft, label='Signal from '+channelArray[2*(zone-1)][0])

				#plt.subplots(212)
				#ax2[1].plot(rangeVec, CH2_Samples, label='Signal from '+channelArray[zone*2-1][0])
				##ax2[1].plot(rangeVecShort, CH2_Env, label='Signal from '+channelArray[zone*2-1][0])
				#ax2[1].plot(freqs, CH2_fft, label='Signal from '+channelArray[2*(zone-1)][0])

				#ax3.plot(tVecShort[CH2_peaks_idx], CH2_detections[CH2_peaks_idx], 'rD')
				ax2[0].legend()
				ax2[1].legend()
				'''
				#plt.tight_layout()

				plt.draw()
				plt.pause(1e-6)

		plt.show()
		TX_freqs = np.linspace(0, fs, int(fs*pulseLength))


		continue
		sig = Sector4_data
		#sig -= lastData
		X = 20.0*np.log10(np.fft.fft(sig))
		RX_freqs = np.linspace(0,fs, len(sig))
		#ax3.plot(RX_freqs, X)#, label=legendtext)

		#X[int(len(X)/2):-1] = 0
		#plt.plot(X, label="X removed samples")
		#Y = np.fft.ifft(X)
		#plt.plot(abs(Y), label='Y')







		#txsig = sig[3319:3319+int(fs*pulseLength)]
		## Plot FFT of matched filter and TX pulse (change idx)
		#plt.plot(freqs,20*np.log10(abs(np.fft.fft(txsig))))
		#plt.plot(freqs,20*np.log10(abs(np.fft.fft(mfilt))))
		#plt.show()

		## Correlating matched filter with RX data ##
		CH1_Out, _ = matchedFilter(Sector4_data, Sector4_data, downSampleStep, mfilt)

		#siginzeros = np.hstack([np.zeros(6591), mfiltham])


		## Plotting all data

		#fig, (ax1, ax2) = plt.subplots(2, figsize=(9,6))
		ax1.plot(rangeVec, Sector4_data,label="Time: "+filename[-18:-5])
		plt.setp(ax1, xlabel='Range [m]')
		plt.setp(ax1, ylabel='Voltage [V]')
		#ax1.set_xlim(0,2.8)
		ax1.set_title("Raw Samples")

		ax2.plot(rangeVec,CH1_Out, label="Time: "+filename[-18:-5])
		plt.setp(ax2, xlabel='Range [m]')
		plt.setp(ax2, ylabel='Correlation')
		#ax2.set_xlim(0,2.8)
		ax2.set_title("Correlated Matched Filter Output")

		ax1.grid()
		ax2.grid()



		#fig2, ax = plt.subplots(1)
		#ax.plot(freqs, 20*np.log10(abs(np.fft.fft(mfilt))), color='red')
		#ax.plot(freqs, 20*np.log10(abs(np.fft.fft(txsig))))
		#ax.plot(mfilt)
		#plt.show()
		#plt.plot(tfilt, mfilt)
		#plt.show()
		#plt.plot(freqs,abs(np.fft.fft(mfilt)))
