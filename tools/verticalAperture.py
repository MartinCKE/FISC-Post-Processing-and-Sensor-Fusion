''' Script for visualizing characterization of vertical beam pattern for FISC.
'''
import numpy as np
import matplotlib.pyplot as plt
import os
import re
from scipy.signal import find_peaks
import datetime
import argparse
from acousticProcessing import gen_mfilt, matchedFilter, butterworth_LP_filter

'''
noShading_1m = [[0.3, '11:11:22'], [0.5, '11:11:27'],
				[0.635, '11:11:34'], [0.77, '11:11:40'],
				[0.89, '11:11:49'], [1.025, '11:11:57'],
				[1.1525, '11:12:04'], [1.28, '11:12:12'],
				[1.4, '11:12:21'], [1.53, '11:12:27'],
				[1.66, '11:12:33'], [1.785, '11:12:38'],
				[1.92, '11:12:45'], [2.06, '11:12:53'],
				[2.18, 'null'], [2.3, 'null']]

noShading_1_5m = [[0.3, '11:16:17'], [0.5, '11:16:25'],
				[0.635, '11:16:32'], [0.77, '11:16:38'],
				[0.89, '11:16:46'], [1.025, '11:16:53'],
				[1.1525, '11:17:00'], [1.28, '11:17:08'],
				[1.4, '11:17:14'], [1.53, '11:17:21'],
				[1.66, '11:17:28'], [1.785, '11:17:36'],
				[1.92, '11:17:42'], [2.06, '11:17:49'],
				[2.18, '11:17:55'], [2.3, '11:18:08']]
noShading_2m = [[0.3, '11:20:00'], [0.5, '11:20:09'],
				[0.635, '11:20:15'], [0.77, '11:20:20'],
				[0.89, '11:20:25'], [1.025, '11:20:31'],
				[1.1525, '11:20:38'], [1.28, '11:20:41'],
				[1.4, '11:20:47'], [1.53, '11:20:52'],
				[1.66, '11:20:57'], [1.785, '11:21:02'],
				[1.92, '11:21:09'], [2.06, '11:21:13'],
				[2.18, '11:21:20'], [2.3, '11:21:25']]

noShading_0_5m = [[0.3, '11:27:24'], [0.5, '11:27:31'],
				[0.635, '11:27:38'], [0.77, '11:27:45'],
				[0.89, '11:27:53'], [1.025, '11:28:00'],
				[1.1525, '11:28:11'], [1.28, '11:28:18'],
				[1.4, '11:28:25'], [1.53, '11:28:31'],
				[1.66, '11:28:37'], [1.785, '11:28:43'],
				[1.92, '11:28:48'], [2.06, '11:28:53'],
				[2.18, 'null'], [2.3, 'null']]

Shading_10mm_1m = [[0.3, '12:05:20'], [0.5, '12:05:26'],
				[0.635, '12:05:33'], [0.77, '12:05:38'],
				[0.89, '12:05:46'], [1.025, '12:05:54'],
				[1.1525, '12:05:58'], [1.28, '12:06:09'],
				[1.4, '12:06:18'], [1.53, '12:06:28'],
				[1.66, '12:06:34'], [1.785, '12:06:40'],
				[1.92, '12:06:45'], [2.06, '12:06:50'],
				[2.18, '12:06:55'], [2.3, '12:06:59']]

Shading_5mm_1m = [[0.3, '12:24:06'], [0.5, '12:24:11'],
				[0.635, '12:24:20'], [0.77, '12:24:25'],
				[0.89, '12:24:31'], [1.025, '12:24:39'],
				[1.1525, '12:24:45'], [1.28, '12:24:50'],
				[1.4, '12:24:55'], [1.53, '12:24:59'],
				[1.66, '12:25:04'], [1.785, '12:25:09'],
				[1.92, '12:25:13'], [2.06, '12:25:17'],
				[2.18, 'null'], [2.3, 'null']]
'''

center = 1.15 ## TX was 1m under surface, and 30cm between TX and RX
d_max = 2.30


def getEchoData(file):
	data=np.load(file, allow_pickle=True)
	acqInfo = data['header']
	imuData = data['IMU']
	O2Data = data['O2']
	fc = int(acqInfo[0])
	BW = int(acqInfo[1])
	pulseLength = acqInfo[2]
	fs = acqInfo[3]
	c = 1463 ## Based on profiler data
	downSampleStep = 1
	if data['sectorData'].ndim == 1 or kwargs.get("sectorFocus"):
		rxdata = data['sectorData'][:]
		nSamples = len(rxdata)


	#sectorFocus = True
	### Acquisition constants ###
	SampleTime = nSamples*(1/fs)
	Range = c*SampleTime/2
	samplesPerPulse = int(fs*pulseLength)  # How many samples do we get per pulse length
	tVec = np.linspace(0, SampleTime, nSamples)
	tVecShort = tVec[0:len(tVec):downSampleStep] # Downsampled time vector for plotting
	plen_d = (c*pulseLength)/2
	rangeVec = np.linspace(-plen_d, Range, len(tVec))
	rangeVecShort = np.linspace(-plen_d, Range, len(tVecShort)).round(decimals=2)
	#rangeVec = np.linspace(0, Range, len(tVec))
	#rangeVecShort = np.linspace(0, Range, len(tVecShort)).round(decimals=2)
	rxdata[0:samplesPerPulse] = 0

	mfilt = gen_mfilt(fc, BW, pulseLength, fs)
	rxdata, _ = matchedFilter(rxdata, rxdata, mfilt, downSampleStep, samplesPerPulse)
	#rxdata -= np.mean(rxdata)

	d=1.6 ## some range shit is wrong with FISC
	range_idx = round((2*d*fs)/c)
	min_idx = range_idx# - 600
	max_idx = range_idx + 200
	#plt.plot(rangeVec[min_idx:max_idx], rxdata[min_idx:max_idx], label=file[-18::])
	#plt.plot(rangeVec,rxdata, label=file[-20::])
	#plt.legend()
	#plt.draw()
	#plt.waitforbuttonpress()
	#plt.pause(0.1)
	#plt.clf()


	#print(file)
	#plt.plot(rangeVec, rxdata)
	#plt.show()
	return rxdata[min_idx:max_idx]

def calcbeam(files, raw, dir):
	beam = []
	angles = []
	startTime = datetime.datetime.strptime(files[0][-18:-4], '%H:%M:%S.%f')
	endTime = datetime.datetime.strptime(files[-1][-18:-4], '%H:%M:%S.%f')
	t = endTime-startTime
	t = t.total_seconds()  ##Time from depth= 0 to 2.3m

	if dir == 'down':
		depths = np.linspace(0, 2.3, len(files)) ## since ball was lowered with constant velocity
	else:
		depths = np.linspace(2.3, 0, len(files)) ## since ball was lowered with constant velocity

	for i, file in enumerate(files):
		data = getEchoData(file) ## matched filter output

		angle = np.arctan((1.15-depths[i]))

		#peaks, _ = find_peaks(data, height=0)
		#data = acousticProcessing.normalizeData(data)
		p = 20*np.log10(data)
		beam.append(np.max(p))
		angles.append(angle)#+np.pi/2)
		#plt.plot(peaks, data[peaks], "x")
		#plt.legend()
		#plt.show()


	fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
	ax.set_theta_zero_location("E")

	if not raw:
		beam = butterworth_LP_filter(beam, cutoff=0.5, fs=25, order=2)
		upper_intersect = np.where(beam>=-3)
		left_upper_intersect = upper_intersect[0][0]
		right_upper_intersect = upper_intersect[0][-1]
		ax.annotate(f" {round(-1*(np.rad2deg(angles[left_upper_intersect]))+np.rad2deg(angles[right_upper_intersect]),2)} deg\nbeam width", xy=(1, beam[left_upper_intersect]-1))


	beam -= np.max(beam) ## normalizing to 0 dB

	ax.set_thetamax(90)
	ax.set_thetamin(-90)
	ax.set_ylabel('dB')
	ax.plot(angles, beam)

	if raw:
		ax.set_title('FISC Total Vertical Beam Pattern, RX_h=10 mm, Unprocessed')
		path = os.getcwd()+'/81358_fisc_post_processing/'
		files = []
		directory = os.path.abspath(os.path.join(path, os.pardir)) +'/plots/'
		plt.savefig(directory+'FISC_VerticalBeams_raw.pdf')
	else:
		ax.set_title('FISC Total Vertical Beam Pattern, RX_h=10 mm, Filtered')
		path = os.getcwd()+'/81358_fisc_post_processing/'
		files = []
		directory = os.path.abspath(os.path.join(path, os.pardir)) +'/plots/'
		plt.savefig(directory+'FISC_VerticalBeams_processed.pdf')

	plt.show()


def loadFileNames(startTime, endTime):#, arr):

	path = os.getcwd()+'/81358_fisc_post_processing/'
	files = []
	directory = os.path.abspath(os.path.join(path, os.pardir)) +'/data/SectorFocus/05-05-22'
	hhmmss_list = []
	i = 0
	for root, dirs, filenames in os.walk(directory, topdown=False):
		for filename in filenames:
			#hhmm = str(re.findall('[0-9]{2}:[0-9]{2}', filename))[2:-2] ## To get HH:MM from filename
			hhmmss = str(re.findall('[0-9]{2}:[0-9]{2}:[0-9]{2}', filename))[2:-2]
			if 'DS' in filename:
				continue
			## Only add desired files to list
			if startTime <= hhmmss <= endTime and filename.endswith('.npz'):
				#print("File added:", filename)
				files.append(root+'/'+filename)

	## Sorting files by time
	files = sorted(files, key=lambda x: x[-18:-4])
	return files

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--raw",
		action="store_true",
		default=False,
		dest='raw',
		help="Raw or filteret processing.",
	)
	args = parser.parse_args()

	startTime = '08:31:37'
	endTime = '08:32:49'
	dir = 'up'
	filenames = loadFileNames(startTime, endTime)#, arr)
	calcbeam(filenames, args.raw, dir)

if __name__ == '__main__':
	main()
