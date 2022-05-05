import numpy as np
import matplotlib.pyplot as plt
import csv
import re
from  matplotlib.colors import LinearSegmentedColormap
from scipy.stats import truncnorm, norm
from matplotlib import cm
from matplotlib.ticker import MaxNLocator
from tools.acousticProcessing import butterworth_LP_filter, polarPlot_init, RX_polarPlot, gen_mfilt, processEcho, peakDetect, colorMapping
from src.IMU import inclination_current

import os
import matplotlib
matplotlib.use('TkAgg')

depthLog_SH = {'09:48':2, '09:49':3, '09:50':4, '09:51':5, '09:52':5}

depthLog_ace = {'12:59:50':1, '12:59:12':2, '12:58:40':3, '12:58:10':4,\
				'12:57:35':5, '12:56:14':6, '12:56:10':7, '12:55:35':8,\
				'12:55:00':9, '12:54:20':10, '12:52:42':11, '12:52:00':12,\
				'12:51:20':13, '12:50:00':14}

def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
	''' From https://stackoverflow.com/questions/36894191/how-to-get-a-normal-distribution-within-a-range-in-numpy '''
	return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

def gaussian(x, mu, sig):
	'''from https://stackoverflow.com/questions/14873203/plotting-of-1-dimensional-gaussian-distribution-function'''
	return 1./(np.sqrt(2.*np.pi)*sig)*np.exp(-np.power((x - mu)/sig, 2.)/2)

def RX_polarPlot_3D(ax, CH1_Intensity, CH2_Intensity, zone, inclinationArr, currentDirArr, headingArr, O2, Temperature, fileName, Z):
	#global rangeLabels, rangeTicks
	sector = zone*2-1

	## Assigning colormap to sampled sector for plotting ##
	timeStamp = str(fileName)[-18:-4]
	date = str(fileName)[-33:-25]

	nBins = len(CH1_Intensity)
	for rangeVal in range(0, nBins):
		colorMap[rangeVal, sector] = CH1_Intensity[rangeVal]
		colorMap[rangeVal, sector+1] = CH2_Intensity[rangeVal]

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

	#axPolar.clear() ## Clearing plot before writing new data



	ax3D.set_title("Ping:"+date+"_"+timeStamp)

	## Polar plot setup ##
	#print("\n Heading:", heading)
	#print("\n")
	ax3D.set_theta_direction(1) #Rotation plotting direction
	ax3D.set_theta_zero_location('N', offset=360-157.5) #Zero location north instead of east. Needs to be set according to PCB mounted orientation!


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

	ax3D.plot_surface(
	X, Y, Z, rstride=1, cstride=1, cmap=plt.get_cmap('jet'),
	linewidth=0, antialiased=False, alpha=0.5)

	## Plotting normalized detector output in corresponding sector ##
	axPolar.scatter(thetaArr_1, rangeBins[CH1_Det_idx], c=CH1_Det[CH1_Det_idx], cmap='RdPu_r', vmin=0, vmax=1) ## Plotting CH1 detections, colormapped
	axPolar.scatter(thetaArr_2, rangeBins[CH2_Det_idx], c=CH2_Det[CH2_Det_idx], cmap='RdPu_r', vmin=0, vmax=1) ## Plotting CH2 detections, colormapped
	plt.draw()


	plt.pause(1e-5) ##



def profilePlot(files, ace):

	## Temp and O2 ranges from literature ##
	tempRange = np.linspace(4,20, 1000)
	o2Range = np.linspace(30, 120, 1000)

	### Function Generation ##
	x_t = np.linspace(-1,1,len(tempRange))
	x_o2 = np.linspace(0,1,len(o2Range))


	a = 7 ## Exponential parameter, trial and error
	y_exp = np.exp(a*x_o2)
	y_exp = (y_exp - np.min(y_exp)) / (np.max(y_exp) - np.min(y_exp)) ## Normalizing
	y_exp = y_exp[int(len(y_exp)/2):-1]# Extracting first half

	y_o2 = np.hstack([y_exp, np.ones((len(y_exp)+2))]) # Full function


	## Normal Distribution for temperature curve ##
	mu = 12.5
	variance = 10
	sigma = np.sqrt(variance)
	y_temps = 1*np.exp(-0.5*((tempRange-mu)/sigma)**2)
	funcfig, (funcax1, funcax2) = plt.subplots(2,1)
	funcax1.plot(tempRange,y_temps)
	funcax1.set_title("Temperature Optimum")
	funcax1.set_xlabel("Temperature [$\degree$C]")
	funcax2.plot(o2Range, y_o2)
	funcax2.set_title("Oxygen Saturation Optimum")
	funcax2.set_xlabel("$\mathrm{O_2}$ Saturation [%]")
	funcfig.suptitle("Aquatic Parameter Optimum Function")
	plt.tight_layout()


	#plt.savefig(os.getcwd()+"/plots/AquaticFunctions.pdf")


	c = ["darkred", "red", "yellow","yellowgreen","lime"]
	#v = [0,.15,.4,.5,0.6,.9,1.]
	v = [0, 0.05, 0.2, 0.6, 1]
	l = list(zip(v,c))
	#cmap=LinearSegmentedColormap.from_list('rg',["r", "w", "g"], N=256)
	custom_cmap=LinearSegmentedColormap.from_list('rg',l, N=256)

	timeData = []

	if ace:
		arr = list(depthLog_ace.items())
	else:
		arr = list(depthLog_SH.items())
	minDepth = 0
	maxDepth = arr[-1][1]

	for i, file in enumerate(files):
		hhmm = str(re.findall('[0-9]{2}:[0-9]{2}', file))[2:-2]
		print("HHMM:", hhmm)
		timeData.append(hhmm)

	times, idx, num_elements = np.unique(timeData,return_counts=True, return_index=True)
	## Removing a few datapoints to map easier
	#del timeData[idx[1]-1], timeData[idx[3]-1],timeData[idx[4]-1]

	times, idx, num_elements = np.unique(timeData,return_counts=True, return_index=True)
	if ace:
		depths = np.linspace(maxDepth,0,len(files))
	else:
		depths = np.linspace(0,maxDepth,len(files))

	temps = []
	O2S = []
	i= 0

	for file in files:
		data=np.load(file, allow_pickle=True)
		O2Data = data['O2']
		temp = np.array_str(O2Data).strip(' []\n')
		dataArr = temp.split(" ")

		if len(dataArr[2]) == 0:
			O2S.append(float(dataArr[3]))
		else:
			O2S.append(float(dataArr[2]))
		#print("O2:", O2S[i])
		i+=1
		temps.append(float(dataArr[-1]))

		'''
		## TEST START ##

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

		if ace:
			c = 1472.5
		else:
			c = 1463

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

		#fig3D = plt.figure()
		#ax3D = fig3D.add_subplot(1,1,1, projection='3d')

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

			figg, axx = plt.subplots(1)
			#axx.plot(echoEnvelope_CH1, label='echoEnvelope_CH1')
			#axx.plot(peaks_CH1, label='peaks_CH1')
			#axx.legend()
			#plt.show()

			detectionArr_CH1[peaks_CH1] = echoEnvelope_CH1[peaks_CH1]
			detectionArr_CH2[peaks_CH2] = echoEnvelope_CH2[peaks_CH2]
			#print(detectionArr_CH1[peaks_CH1])
			#print(detectionArr_CH2[peaks_CH2])

			maxPeak_CH1 = np.max(echoEnvelope_CH1[peaks_CH1])
			maxPeak_CH2 = np.max(echoEnvelope_CH2[peaks_CH2])

			maxPeak_idx_CH1 = np.argmax(echoEnvelope_CH1[peaks_CH1])
			maxPeak_idx_CH2 = np.argmax(echoEnvelope_CH2[peaks_CH2])
			## Extract distance to peak ##
			dist_CH1 = rangeVec[peaks_CH1][maxPeak_idx_CH1]
			dist_CH2 = rangeVec[peaks_CH2][maxPeak_idx_CH2]


			#RX_polarPlot_3D(ax3D, CH1_Intensity, CH2_Intensity, zone, inclinationArr, currentDirArr, headingArr, O2S[i-1], temps[i-1], file, depths[i-1])




			RX_polarPlot(CH1_Intensity, CH2_Intensity, zone, [0,0,0], [0,0,0], \
						inclinationArr, currentDirArr, headingArr, O2S[i-1], temps[i-1], file, sectorFocus=False)
		#plt.show()

		## TEST END ##
		'''



	O2S_filtered = butterworth_LP_filter(O2S, cutoff=2, fs=10, order=2)

	if ace:
		temps_filtered = butterworth_LP_filter(temps, cutoff=2, fs=20, order=2)
	else:
		temps_filtered = butterworth_LP_filter(temps, cutoff=2, fs=10, order=2)
	#temps_filtered  = temps
	#O2S_filtered = O2S
	### Meshgrid ###
	meshfig, meshax = plt.subplots(1, figsize=(9,7))
	O2Mesh, TempMesh = np.meshgrid(y_o2, y_temps, indexing='xy')
	Z = O2Mesh*TempMesh
	#Z = (O2Mesh+TempMesh)/2
	plt.imshow(Z, cmap=custom_cmap, aspect='auto', origin='lower',extent=[min(o2Range), max(o2Range), min(tempRange), max(tempRange)])
	#extent=[min(o2Range), max(o2Range), min(tempRange), max(tempRange)]

	#meshax.set_xticks(np.linspace(min(o2Range), max(o2Range), 6))
	#meshax.yaxis.set_major_locator(MaxNLocator(10))
	### extent=[o2Range[0],o2Range[-1],tempRange[0],tempRange[-1]]
	plt.locator_params(axis='y', nbins=10)
	plt.locator_params(axis='x', nbins=10)
	#xtick_list = ['x', '30', '39', '48', '57', '66', '75', '84', '93', '102', '111', '120', '12']
	#ytick_list = ['4', '5.6', '7.2', '8.8', '10.4', '12', '13.6', '15.21', '16.8', '18.4', '20']
	#meshax.set_xticklabels(xtick_list)
	#meshax.set_yticklabels(ytick_list)

	meshax.set_title("Meshgrid for Aquatic Parameter Optimum")
	cbar = plt.colorbar()
	cbar.ax.set_ylabel('Aquatic Environment Score Bar')#, rotation=270)
	plt.xlabel("$\mathrm{O_2}$ Saturation [%]")
	plt.ylabel("Temperature [$\degree$C]")
	#plt.savefig(os.getcwd()+"/plots/AquaticMeshgrid_AltHeatmap.pdf")
	#plt.show()

	## Assigning color to points ##
	colors = []
	colors_temp = []
	colors_o2 = []



	for i, val in enumerate(O2S_filtered):
		t = temps_filtered[i]
		o2 = O2S_filtered[i]
		print("t:", t)
		print("O2:", o2)
		print("Depth:", depths[i])

		t_idx = np.where(tempRange > t)[0][0]
		o2_idx = np.where(o2Range > o2)[0][0]
		print("\n\r t_idx:", t_idx)
		print("temp[idx]", tempRange[t_idx])
		print("o2idx:", o2_idx)
		print("o2[idx]", o2Range[o2_idx])

		print('Z[o2_idx][t_idx]', Z[o2_idx][t_idx])

		t_val = y_temps[np.where(tempRange > t)][0]
		o2_val = y_o2[np.where(o2Range > o2)][0]

		testidx = np.where(tempRange > 10)[0][0]
		#print("TEST:", Z[999,500])

		Zval = Z[o2_idx, t_idx]

		row, col = np.where(Z)
		meshax.plot(o2Range[o2_idx], tempRange[t_idx], 'x', color='black')
		#print('oubbiu', O2Mesh[-1])
		#meshax.plot(Z[o2_idx][t_idx], 'x')
		#meshax.plot(o2_idx, t_idx, 'x', color='blue')
		#print(np.where())


		print("tval:", t_val)
		print("o2val:", o2_val)
		totval = t_val*o2_val
		print("totval:", totval)
		print("Zval:", Zval)
		colors.append((t_val+o2_val)/2) ## Avg value
		#colors.append(totval) ## Multiplicative, more strict
		colors_temp.append(t_val)
		colors_o2.append(o2_val)
		#quit()

	#c = colors
	#plt.scatter(colors_temp, colors_o2, c=c,
            #cmap = custom_cmap, alpha =0.5)
	#cbar = plt.colorbar()
	#cbar.set_label('Color Intensity')

	#print("Colors:", colors)
	#print("O2 colors:", colors_o2)
	#ax1.plot(O2S, depths, alpha=0.5)

	profilefig, (ax1, ax2, ax3) = plt.subplots(1,3,sharey=True, figsize=(10,7))

	ax1.plot(O2S_filtered, depths, color='black', alpha=0.5)
	ax1.scatter(O2S_filtered,depths, c=colors_o2, cmap=custom_cmap, edgecolor='none', vmin=0, vmax=1)


	ax2.plot(temps_filtered, depths, color='black', alpha=0.5)
	ax2.scatter(temps_filtered, depths, c=colors_temp, cmap=custom_cmap, edgecolor='none', vmin=0, vmax=1)

	ax3.plot(colors, depths, color='black', alpha=0.5)
	ax3.scatter(colors, depths, c=colors, cmap=custom_cmap, edgecolor='none', vmin=0, vmax=1)
	ax3.set_title('Combined Water Quality Score')
	ax3.set_xlabel('Score $\in$[0,1]')
	ax3.set_xlim([0,1])
	#.colorbar(Z)
	#cbar = plt.colorbar(custom_cmap)
	#cbar.set_label('Color Intensity')
	#im = ax1.imshow(colors, cmap=custom_cmap)
	#fig.colorbar(im, cax=cax, orientation='vertical')
	#plt.colorbar()
	plt.gca().invert_yaxis()
	#plt.locator_params(axis='y', nbins=6)
	#plt.locator_params(axis='x', nbins=10)
	ax1.xaxis.set_major_locator(plt.MaxNLocator(7))
	ax1.yaxis.set_major_locator(plt.MaxNLocator(10))
	ax2.xaxis.set_major_locator(plt.MaxNLocator(6))
	ax2.yaxis.set_major_locator(plt.MaxNLocator(10))
	ax1.set_title('$\mathrm{O_2}$ Saturation Profile')
	ax1.set_xlabel('$\mathrm{O_2}$ Saturation [%]')
	ax2.set_title('Water Temperature Profile')
	ax2.set_xlabel('Temperature [$^\circ$C]')
	ax1.set_ylabel('Depth [m]')
	ax1.grid()
	ax2.grid()
	ax3.grid()
	if ace:
		profilefig.suptitle('Profile Measurements at Rataren 2')
		#plt.savefig(os.getcwd()+"/plots/O2_temp_profile_ACE_AltHeatmap.pdf")
	else:
		profilefig.suptitle('Profile Measurements at Sinkaberg Hansen (Rørvik)')
		#plt.savefig(os.getcwd()+"/plots/O2_temp_profile_SH_AltHeatmap.pdf")




	plt.show()

	quit()
	#data=np.load(file, allow_pickle=True)
	#acqInfo = data['header']
	#imuData = data['IMU']
	#O2Data = data['O2']


def profilePlot_old(files, ace):
	timeData = []
	minDepth = 0
	maxDepth = 5
	for i, file in enumerate(files):
		hhmm = str(re.findall('[0-9]{2}:[0-9]{2}', file))[2:-2]
		print("HHMM:", hhmm)
		timeData.append(hhmm)
	print(timeData)
	quit()
	times, idx, num_elements = np.unique(timeData,return_counts=True, return_index=True)
	## Removing a few datapoints to map easier
	del timeData[idx[1]-1], timeData[idx[3]-1],timeData[idx[4]-1]

	times, idx, num_elements = np.unique(timeData,return_counts=True, return_index=True)
	depths = np.linspace(0,5,len(files))

	temps = []
	O2S = []
	for file in files:
		data=np.load(file, allow_pickle=True)
		O2Data = data['O2']
		temp = np.array_str(O2Data).strip(' []\n')
		dataArr = temp.split(" ")

		if len(dataArr[2]) == 0:
			O2S.append(float(dataArr[3]))
		else:
			O2S.append(float(dataArr[2]))

		temps.append(float(dataArr[-1]))

	fig2, (ax1, ax2) = plt.subplots(1,2,sharey=True)

	O2S_filtered = acousticProcessing.butterworth_LP_filter(O2S, cutoff=2, fs=10, order=2)
	temps_filtered = acousticProcessing.butterworth_LP_filter(temps, cutoff=2, fs=10, order=2)

	ax1.plot(O2S, depths, alpha=0.5)
	ax1.plot(O2S_filtered, depths, color='black', alpha=0.5)
	ax2.plot(temps, depths)
	ax2.plot(temps_filtered, depths, color='black', alpha=0.5)
	plt.gca().invert_yaxis()
	#plt.locator_params(axis='y', nbins=6)
	#plt.locator_params(axis='x', nbins=10)
	ax1.xaxis.set_major_locator(plt.MaxNLocator(7))
	ax1.yaxis.set_major_locator(plt.MaxNLocator(10))
	ax2.xaxis.set_major_locator(plt.MaxNLocator(6))
	ax2.yaxis.set_major_locator(plt.MaxNLocator(10))
	ax1.set_title('O2 Saturation Profile')
	ax1.set_xlabel('O2 Saturation [%]')
	ax2.set_title('Water Temperature Profile')
	ax2.set_xlabel('Water Temprature [C$^\circ$]')
	ax1.set_ylabel('Depth [m]')
	ax1.grid()
	ax2.grid()

	plt.show()
	quit()
	#data=np.load(file, allow_pickle=True)
	#acqInfo = data['header']
	#imuData = data['IMU']
	#O2Data = data['O2']

def O2TempPlot(files):
	timeData = []

	for i, file in enumerate(files):
		hhmm = str(re.findall('[0-9]{2}:[0-9]{2}', file))[2:-2]
		print("HHMM:", hhmm)
		timeData.append(hhmm)

	times, idx, num_elements = np.unique(timeData,return_counts=True, return_index=True)


	times, idx, num_elements = np.unique(timeData,return_counts=True, return_index=True)

	temps = []
	O2S = []
	for file in files:
		data=np.load(file, allow_pickle=True)
		O2Data = data['O2']
		temp = np.array_str(O2Data).strip(' []\n')
		dataArr = temp.split(" ")

		if len(dataArr[2]) == 0:
			O2S.append(dataArr[3])
		else:
			O2S.append(dataArr[2])

		temps.append(dataArr[-1])

	fig2, (ax1, ax2) = plt.subplots(2)
	ax1.plot(timeData, O2S)
	ax2.plot(timeData, temps)

	#plt.locator_params(axis='y', nbins=6)
	#plt.locator_params(axis='x', nbins=10)

	ax1.xaxis.set_major_locator(plt.MaxNLocator(30))
	ax1.yaxis.set_major_locator(plt.MaxNLocator(30))
	ax2.xaxis.set_major_locator(plt.MaxNLocator(30))
	ax2.yaxis.set_major_locator(plt.MaxNLocator(30))
	'''
	ax1.set_title('O2 Saturation Profile')
	ax1.set_xlabel('O2 Saturation [%]')
	ax2.set_title('Water Temperature Profile')
	ax2.set_xlabel('Water Temprature [C$^\circ$]')
	ax1.set_ylabel('Depth [m]')
	ax1.grid()
	ax2.grid()
	'''

	plt.show()
	quit()


def SVProfilePlot(ace):
	''' Function for generating sound velocity profile plot from Norbit SVP CVS file.
	'''
	data = []

	if ace:
		with open("data/026157_2022-03-11_12-46-56.csv") as file:
			file_reader = csv.reader(file)
			l = 0
			for row in file_reader:
				if len(row) < 5:
					continue
				data.append(row)
		del data[0]
	else:
		with open("data/026157_2022-02-16_13-55-49.csv") as file:
			file_reader = csv.reader(file)
			l = 0
			for row in file_reader:
				if len(row) < 5:
					continue
				data.append(row)
		del data[0]

	c = [float(data[i][2]) for i, val in enumerate(data)]
	p = [float(data[i][3]) for i, val in enumerate(data)]
	del c[0]
	del p[0]

	fig, ax = plt.subplots(1)
	plt.plot(c, p)
	#plt.style.use('ggplot')
	#plt.style.use('dark_background')
	plt.gca().invert_yaxis()
	plt.xlabel('SV [m/s]')
	plt.ylabel('Depth [m]')
	plt.grid()
	ax.set_xlim([min(c)-2,max(c)+2])
	if ace:
		plt.title('Sound Velocity Profile at Rateren II')
		#plt.savefig(os.getcwd()+'/plots/SV_profile_ACE.pdf')
	else:
		plt.title('Sound Velocity Profile at Rørvik (Sinkaberg Hansen)')
		#plt.savefig(os.getcwd()+'/plots/SV_profile_SH.pdf')
