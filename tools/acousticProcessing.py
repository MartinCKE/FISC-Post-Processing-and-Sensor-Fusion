import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfilt, sosfreqz, filtfilt
from pylab import *

def processEcho(data, fc, BW, pulseLength, fs, downSampleStep, samplesPerPulse, min_idx):
	''' Function for processing echo data from one sector.
		Input: Raw ADC samples from RX.
		Output:
	'''
	## Generate matched filter from TX pulse parameters ##
	mfilt = gen_mfilt(fc, BW, pulseLength, fs)
	## Fetch matched filter output envelope ##
	echoData_Env, _ = matchedFilter(data, data, mfilt, downSampleStep, samplesPerPulse) ## Extracting matched filter envelope
	## Add digital TVG, bypassed as of right now ##
	#echoData_Env = TVG(echoData_Env, Range, c, fs) ## Adding digital TVG

	## Remove TX pulse noise from start ##
	#echoData_Env[0:samplesPerPulse] = 1 ## To remove tx pulse noise

	## Using CA-CFAR to detect peaks ##
	if downSampleStep == 1:
		CH1_peaks_idx, CH1_noise, CH1_detections, CH1_thresholdArr = peakDetect(echoData_Env, num_train=80, num_guard=10, rate_fa=0.3)
	else:
		CH1_peaks_idx, CH1_noise, CH1_detections, CH1_thresholdArr = peakDetect(echoData_Env, num_train=3, num_guard=5, rate_fa=0.3)

	print("Data length:", len(echoData_Env))
	print("Detections length:", len(CH1_detections))
	## Using "scipy find peaks" to extract one detection per peak
	#CH1_detections = normalizeData(CH1_detections)
	CH1_detections = CH1_detections - 1 ## Removing offset

	threshold = np.max(CH1_detections)/3 ## Thresholding

	peaks_idx, _ = scipy.signal.find_peaks(CH1_detections, distance=5, height=threshold)

	peaks_idx = np.delete(peaks_idx, np.where(peaks_idx < min_idx)) ## Removing peak detections closer than min range
	'''
	print("Detections:", CH1_detections)
	fig, ax = plt.subplots(1)
	ax.plot(CH1_detections, label='data input')
	#ax.plot(test, color='black')
	ax.plot(echoData_Env, label='Signal envelope')
	plt.plot(peaks_idx, CH1_detections[peaks_idx], "x", label='Detections')
	plt.plot(np.zeros_like(CH1_detections), "--", color="gray", label='threshold')
	plt.legend()
	plt.show()
	quit()
	'''


	return echoData_Env, peaks_idx

def thresholding_algo(y, lag, threshold, influence):
	signals = np.zeros(len(y))
	filteredY = np.array(y)
	avgFilter = [0]*len(y)
	stdFilter = [0]*len(y)
	avgFilter[lag - 1] = np.mean(y[0:lag])
	stdFilter[lag - 1] = np.std(y[0:lag])
	for i in range(lag, len(y)):
		if abs(y[i] - avgFilter[i-1]) > threshold * stdFilter [i-1]:
			if y[i] > avgFilter[i-1]:
				signals[i] = 1
			else:
				signals[i] = -1

			filteredY[i] = influence * y[i] + (1 - influence) * filteredY[i-1]
			avgFilter[i] = np.mean(filteredY[(i-lag+1):i+1])
			stdFilter[i] = np.std(filteredY[(i-lag+1):i+1])
		else:
			signals[i] = 0
			filteredY[i] = y[i]
			avgFilter[i] = np.mean(filteredY[(i-lag+1):i+1])
			stdFilter[i] = np.std(filteredY[(i-lag+1):i+1])

	return dict(signals = np.asarray(signals),
				avgFilter = np.asarray(avgFilter),
				stdFilter = np.asarray(stdFilter))

def butterworth_LP_filter(data, cutoff, fs, order):
	nyq = 0.5 * fs
	normal_cutoff = cutoff / nyq
	#Filter coefficients
	b, a = butter(order, normal_cutoff, btype='low', analog=False)
	y = filtfilt(b, a, data)
	return y

def butterworth_BP_filter(data, lowcut, highcut, fs, order=3):
		nyq = 0.5 * fs
		low = lowcut / nyq
		high = highcut / nyq
		sos = butter(order, [low, high], analog=False, btype='band', output='sos')
		w, h = scipy.signal.sosfreqz(sos)
		y = sosfilt(sos, data)
		'''
		plt.plot(data, color='red')
		plt.plot(y, color='black')
		#plt.plot((fs * 0.5 / np.pi) * w, abs(h), label="order = %d" % order)

		#db = 20*np.log10(np.abs(h))
		#plt.plot(w/np.pi, db, 'orange', label='')

		plt.show()
		quit()
		'''
		return y

def TVG(data, Range, c, fs):

	times = np.linspace(0,Range/c, len(data))

	ranges = c*times
	gainArr = np.zeros((len(ranges)))
	print(ranges)

	for i, val in enumerate(ranges):
		if ranges[i] < 1:
			ranges[i] = 20.0*np.log10(1)#1
		else:
			gainArr[i] = 20.0*np.log10(ranges[i])

	#gainArr = G_pre + 20*np.log10(ranges)

	#plt.plot(ranges, gainArr, color='red', alpha=0.5)
	data_gained = gainArr+20.0*np.log10(data)
	'''
	plt.plot(ranges, data_gained, color='red')
	test = 10**(gainArr/20)
	test = 20*np.log10(data*test)
	plt.plot(ranges, test, color='orange')
	plt.show()
	fig1, ax1 = plt.subplots(1)
	ax1.plot(ranges, 20.0*np.log10(data))
	plt.show()
	quit()
	'''



	return data_gained*10**(data_gained/20)

def gen_mfilt(fc, BW, pulseLength, fs):
	### Matched filter w/ Hamming window ###
	tfilt = np.linspace(0, pulseLength, int(fs*pulseLength))
	mfilt = scipy.signal.chirp(tfilt, int(fc-BW/2), tfilt[-1], int(fc+BW/2),method='linear',phi=90)
	mfilt = mfilt*np.hamming(len(mfilt))*1.85

	return mfilt

def normalizeData(data):
	'''
	figg, axx = plt.subplots(1)
	axx.plot(data, color='red')
	data[data<0] = 0
	axx.plot(data, color='black')
	plt.show()
	'''
	if not np.all((data == 0)):
		return (data - np.min(data)) / (np.max(data) - np.min(data))
	else:
		print('Only zeros ecountered, check data.')
		return data

def peakDetect(data, num_train=6, num_guard=2, rate_fa=1e-3):
	''' Cell-Averaging Constant False Alarm Rate (CFAR) detector algorithm.

		CUT = Cell under test
		Parameters:
		num_train = N samples surrounding CUT, assumed to be noise
		num_guard = N samples adjacent to CUT to avoid signal leakage to noise
		rate_fa = chosen false alarm rate (default = 0.001)
	'''
	#data = normalizeData(data)
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

		#alpha = 1.2# ## Custom threshold (rate_fa ignored)
		threshold = alpha * p_noise
		thresholdArr.append(threshold)

		if data[i] > threshold:
			peak_idx.append(i)
			peaks.append(p_noise)
			if peaks[i] <= 0:
				peaks[i] = 0.00001

		else:
			peaks.append(data[i])
			if peaks[i] <= 0:
				peaks[i] = 0.00001

	noiseArr.append(p_noise)
	peak_idx = np.array(peak_idx, dtype=int)

	detectorarr = data/peaks
	#detectorarr = normalizeData(detectorarr)
	'''
	fig, ax = plt.subplots(1)
	ax.plot(data, color='red', label='data', alpha=0.5)
	ax.plot(noiseArr, color='black', label='noise', alpha=0.5)
	ax.plot(peaks, color='magenta', label='peaks', alpha=0.5)
	ax.plot(thresholdArr, color='blue', label='threshold', alpha=0.5)
	ax.plot(detectorarr, color='black', label='detectorarr')
	plt.legend()
	plt.show()
	#quit()
	'''


	return peak_idx, noiseArr, detectorarr, thresholdArr


def Hilbert(CH1_Data, CH2_Data, downSampleStep):
	''' Does basic frequency-domain analysis
		as well as downsampling the envelope time signal for
		making plotting faster. Will be used in more technical
		signal analysis later on.
	'''
	### Hilbert transform on signal ###
	CH1_Hilb = scipy.signal.hilbert(CH1_Data)
	CH2_Hilb = scipy.signal.hilbert(CH2_Data)

	### Acquiring time-domain envelope signal and downsampling it ###
	CH1_Env = abs(CH1_Hilb)
	CH1_EnvShort = CH1_Env[0:len(CH1_Env):downSampleStep]
	CH2_Env = abs(CH2_Hilb)
	CH2_EnvShort = CH2_Env[0:len(CH2_Env):downSampleStep]

	### Get phase of signal ##
	CH1_Phase = np.unwrap(np.angle(CH1_Hilb))
	CH2_Phase = np.unwrap(np.angle(CH2_Hilb))

	### Acquiring time domain "envelope" of frequency content ###
	CH1_Freq = np.diff(CH1_Phase) / (2*np.pi) * fs
	CH2_Freq = np.diff(CH2_Phase) / (2*np.pi) * fs


	return CH1_EnvShort, CH2_EnvShort, CH1_Freq, CH2_Freq

def matchedFilter(CH1_data, CH2_data, mfilt, downSampleStep, samplesPerPulse):
	''' Correlate data from 2 channels with pulse replica.
		Returns envelope of correlation.
	'''
	#fig2, ax2 = plt.subplots(1)
	CH1_data = CH1_data-np.mean(CH1_data)
	CH2_data = CH2_data-np.mean(CH2_data)
	#ax2.plot(CH1_data, label='after sub')

	CH1_corr = scipy.signal.correlate(CH1_data, mfilt, mode='same', method='fft')
	CH2_corr = scipy.signal.correlate(CH2_data, mfilt, mode='same', method='fft')

	CH1_Env = (abs(scipy.signal.hilbert(CH1_corr)))
	CH2_Env = (abs(scipy.signal.hilbert(CH2_corr)))
	CH1_Env[0:samplesPerPulse] = 1
	CH2_Env[0:samplesPerPulse] = 1

	CH1_EnvShort = CH1_Env[0:len(CH1_Env):downSampleStep]
	CH2_EnvShort = CH2_Env[0:len(CH2_Env):downSampleStep]


	return CH1_EnvShort, CH2_EnvShort

def polarPlot_init(tVecShort, rangeVecShort):
	''' Initializes empty polar plot for data visualization.
		Plots downsampled data based on setting in main.py.
	'''
	global nBins, theta, rangeBins, colorMap, fig, axPolar, rangeLabels, rangeTicks

	### Polar plot setup ###
	nBins = len(tVecShort)
	theta = np.array([(np.pi/4)*n for n in range(9)]) # 8 sectors, each pi/4 large
	rangeBins = np.array(range(nBins)) # Range bins 0-nBins (bin resolution in plot)
	colorMap = np.zeros((nBins,9)) ## Make empty colormap matrix for intensity

	fig1 = plt.figure(figsize=(8,7))
	axPolar = plt.subplot(111, projection='polar') ## Make polar plot
	axPolar.grid(True)
	axPolar.margins(y=0)
	fig1.suptitle("FISC Acoustic Plot")

	#axPolar.set_rmax(Range+1) # test
	rangeTicks  = np.linspace(0, nBins-1, 5, dtype=int)
	rangeLabels = rangeVecShort[rangeTicks]

def colorMapping(CH1_data, CH2_data):
	''' Maps the time domain envelope voltage
		to colors.
		Essentially B-scan intensity color mapping.
	'''

	CH1_colorMap = np.zeros(len(CH1_data))
	CH2_colorMap = np.zeros(len(CH2_data))
	## Max/Min values for mapping ##
	## Voltage levels in ##
	inMin = 0
	inMax = 10 ## Color sensitivity setting
	## Colormap range ##
	outMin = 0
	outMax = 1
	#hm, ax5 =plt.subplots(1)
	#ax5.plot(CH1_data, color='red')
	for i, val in enumerate(CH1_data):
		CH1_colorMap[i] = (val-inMin) * (outMax-outMin) / (inMax-inMin) + outMin

	for i, val in enumerate(CH2_data):
		CH2_colorMap[i] = (val-inMin) * (outMax-outMin) / (inMax-inMin) + outMin
	#ax5.plot(CH1_colorMap, color='black')
	#plt.show()
	#quit()
	return CH1_colorMap, CH2_colorMap

def RX_polarPlot(CH1_Intensity, CH2_Intensity, zone, CH1_Det, CH2_Det, inclination, currentDir, heading, O2, Temp, fileName, sectorFocus=False):
	''' Plots the received echo intensities on a polar plot with heading information.
	'''
	global rangeLabels, rangeTicks
	sector = zone*2-1

	## Assigning colormap to sampled sector for plotting ##
	if sectorFocus:
		timeStamp = str(fileName)[-18:-4]
		date = str(fileName)[-45:-37]
		for rangeVal in range(0, nBins):
			colorMap[rangeVal, sector+1] = CH1_Intensity[rangeVal]
	else:
		timeStamp = str(fileName)[-18:-4]
		date = str(fileName)[-33:-25]
		for rangeVal in range(0, nBins):
			colorMap[rangeVal, sector] = CH1_Intensity[rangeVal]
			colorMap[rangeVal, sector+1] = CH2_Intensity[rangeVal]

	#figg, axx = plt.subplots(1)
	#axx.plot(CH2_Intensity)
	#print(CH2_Intensity)
	#plt.show()
	#quit()
	TH = cbook.simple_linear_interpolation(theta, 5) ## Rounding bin edges

	##Properly padding out C so the colors go with the right sectors
	#start[0] = time.time()
	C = np.zeros((rangeBins.size, TH.size))
	oldfill = 0
	TH_ = TH.tolist()
	for i in range(theta.size):
		fillto = TH_.index(theta[i])
		for j, x in enumerate(colorMap[:,i]):
			C[j, oldfill:fillto].fill(x)
		oldfill = fillto

	axPolar.clear() ## Clearing plot before writing new data



	axPolar.set_title("Ping:"+date+"_"+timeStamp)

	## Polar plot setup ##
	#print("\n Heading:", heading)
	#print("\n")
	axPolar.set_theta_direction(1) #Rotation plotting direction
	axPolar.set_theta_zero_location('N', offset=360-157.5) #Zero location north instead of east. Needs to be set according to PCB mounted orientation!
	#axPolar.set_theta_offset(np.deg2rad(heading)) #Rotating plot with compass heading

	#northArrow = np.full((20,), np.deg2rad(heading+157.5))

	if type(heading) is list:
		northArrow = np.full((50,), np.deg2rad(heading[-1]+157.5))
		eastArrow = np.full((50,), np.deg2rad(heading[-1]+157.5+90))
		westArrow = np.full((50,), np.deg2rad(heading[-1]+157.5-90))
		southArrow = np.full((50,), np.deg2rad(heading[-1]+157.5+180))
	else:
		northArrow = np.full((50,), np.deg2rad(heading+157.5))
		eastArrow = np.full((50,), np.deg2rad(heading+157.5+90))
		westArrow = np.full((50,), np.deg2rad(heading+157.5-90))
		southArrow = np.full((50,), np.deg2rad(heading+157.5+180))

	r = np.arange(rangeBins[-1]-50, rangeBins[-1])
	axPolar.plot(northArrow, r, color='red')
	axPolar.plot(eastArrow, r, color='white')
	axPolar.plot(southArrow, r, color='white')
	axPolar.plot(westArrow, r, color='white')


	## Setting range and theta ticks/labels ##
	#axPolar.set_xticks(np.arange(0,2.0*np.pi,np.pi/4.0))
	#axPolar.set_xticklabels(['SSE', 'ESE', 'ENE', 'NNE', 'NNW', 'WNW', 'WSW', 'SSW'])
	axPolar.set_xticklabels([])
	axPolar.set_yticks(rangeTicks)
	axPolar.set_yticklabels(rangeLabels, fontsize=12) #Range labels in meters
	axPolar.tick_params(colors='red')

	for i in range(1,9):
		## Adding Sector identifier text ##
		thetaPos = 2*np.pi*i/8 - np.pi/8 + 0.1
		axPolar.text(thetaPos-0.05, rangeBins[-100], "Sector "+str(i),bbox=dict(facecolor='red', alpha=0.4))

	axPolar.text(0.065, 0.05, 'Vertical inclination:'+str(inclination)+" (degrees)", transform=plt.gcf().transFigure)
	axPolar.text(0.02, 0.02, 'Water current direction:'+str(currentDir)+" (degrees)", transform=plt.gcf().transFigure)
	axPolar.text(0.073, 0.08, 'Heading direction:'+str(heading)+"(degrees)", transform=plt.gcf().transFigure)

	axPolar.text(0.65, 0.08, 'Water $\mathrm{O_2}$ Saturation: '+str(O2)+" %", transform=plt.gcf().transFigure)
	axPolar.text(0.65, 0.02, 'Water Temperature: '+str(Temp)+" (degrees)", transform=plt.gcf().transFigure)

	'''
	axPolar.text(0, 0, "test",
      horizontalalignment='left',
      verticalalignment='top',
      size='large',
      bbox=dict(facecolor='red', alpha=1.0),
      transform=plt.gca().transAxes)
	 '''
	## Plotting meshgrid ##
	th, r = np.meshgrid(TH, rangeBins)
	C = normalizeData(C) ## Normalizing colormap array from 0 to 1
	axPolar.pcolormesh(th, r, C, cmap='cividis', shading='gouraud', vmin=0, vmax=1)# shading='gouraud' gives smoothing
	axPolar.grid()

	## Normalizing detector output from 0 to 1 ##
	CH1_Det = normalizeData(CH1_Det)
	CH2_Det = normalizeData(CH2_Det)
	#print("detections:", CH1_Det)
	#print(len(rangeBins))
	#print("sector:", sector)
	#print("bl:",(sector*2 - 1))

	CH1_Det_idx = np.asarray(np.where(CH1_Det > 0.0)) ## To only plot actual detections
	CH2_Det_idx = np.asarray(np.where(CH2_Det > 0.0)) ## To only plot actual detections

	thetaArr_1 = np.full((CH1_Det_idx.shape[1]), (sector*2 - 1)*np.pi/8)
	thetaArr_2 = np.full((CH2_Det_idx.shape[1]), ((sector+1)*2 - 1)*np.pi/8)

	## Plotting normalized detector output in corresponding sector ##
	axPolar.scatter(thetaArr_1, rangeBins[CH1_Det_idx], c=CH1_Det[CH1_Det_idx], cmap='RdPu_r', vmin=0, vmax=1) ## Plotting CH1 detections, colormapped
	axPolar.scatter(thetaArr_2, rangeBins[CH2_Det_idx], c=CH2_Det[CH2_Det_idx], cmap='RdPu_r', vmin=0, vmax=1) ## Plotting CH2 detections, colormapped
	plt.draw()


	plt.pause(1e-5) ##
