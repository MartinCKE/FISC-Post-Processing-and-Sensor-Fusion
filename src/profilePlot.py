import numpy as np
import matplotlib.pyplot as plt
import csv
import re
from  matplotlib.colors import LinearSegmentedColormap
from scipy.stats import truncnorm, norm
from matplotlib import cm
from matplotlib.ticker import MaxNLocator
from tools.acousticProcessing import butterworth_LP_filter
import os


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
	#plt.savefig(os.getcwd()+"/plots/AquaticMeshgrid.pdf")
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
		#colors.append(totval)
		colors.append(totval)
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
		plt.savefig(os.getcwd()+"/plots/O2_temp_profile_ACE.pdf")
	else:
		profilefig.suptitle('Profile Measurements at Sinkaberg Hansen (Rørvik)')
		plt.savefig(os.getcwd()+"/plots/O2_temp_profile_SH.pdf")




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
