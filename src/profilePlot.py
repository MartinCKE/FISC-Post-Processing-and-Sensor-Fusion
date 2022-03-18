import numpy as np
import matplotlib.pyplot as plt
import csv
import re

from tools.acousticProcessing import butterworth_LP_filter


depthLog_SH = {'09:48':2, '09:49':3, '09:50':4, '09:51':5, '09:52':5}

depthLog_ace = {'12:59:50':1, '12:59:12':2, '12:58:40':3, '12:58:10':4,\
                '12:57:35':5, '12:56:14':6, '12:56:10':7, '12:55:35':8,\
                '12:55:00':9, '12:54:20':10, '12:52:42':11, '12:52:00':12,\
                '12:51:20':13, '12:50:00':14}


def profilePlot(files, ace):
    timeData = []

    if ace:
        arr = list(depthLog_ace.items())
    else:
        arr = list(depthLog_SH.items())
    minDepth = 0
    maxDepth = arr[-1][1]
    #print(maxDepth)
    #quit()

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
        print("file", file)
        print("depth:", depths[i])


        if len(dataArr[2]) == 0:
            O2S.append(float(dataArr[3]))
        else:
            O2S.append(float(dataArr[2]))
        print("O2:", O2S[i])
        i+=1
        temps.append(float(dataArr[-1]))
    #quit()
    fig2, (ax1, ax2) = plt.subplots(1,2,sharey=True)
    ax1.plot(O2S, color='red')
    plt.show()
    O2S_filtered = butterworth_LP_filter(O2S, cutoff=2, fs=10, order=2)
    
    if ace:
        temps_filtered = butterworth_LP_filter(temps, cutoff=2, fs=20, order=2)
    else:
        temps_filtered = butterworth_LP_filter(temps, cutoff=2, fs=10, order=2)

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
    plt.title('Sound Velocity Profile')
    plt.xlabel('SV [m/s]')
    plt.ylabel('Depth [m]')
    plt.grid()
    ax.set_xlim([min(c)-2,max(c)+2])
    #plt.show()
