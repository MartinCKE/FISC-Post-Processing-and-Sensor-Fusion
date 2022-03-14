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
import fusePlot
import acousticProcessing
import profilePlot
import IMU


matplotlib.use('TkAgg')
channelArray = [['CH1', '0'],['CH2', '2'],
                ['CH3', '2'],['CH4', '0'],
                ['CH5', '1'],['CH6', '3'],
                ['CH7', '3'],['CH8', '1']]


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

    fig1 = plt.figure(figsize=(6,6))
    axPolar = plt.subplot(111, projection='polar') ## Make polar plot
    axPolar.grid(True)
    axPolar.margins(y=0)
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


def RX_polarPlot_OLD(CH1_Intensity, CH2_Intensity, zone, heading, CH1_Det, CH2_Det):
    ''' Plots the received echo intensities on a polar plot
    '''
    global rangeLabels, rangeTicks
    sector = zone*2-1

    for rangeVal in range(0, nBins):
        colorMap[rangeVal, sector+1] = CH1_Intensity[rangeVal]
        colorMap[rangeVal, sector] = CH2_Intensity[rangeVal]

    TH = cbook.simple_linear_interpolation(theta, 5) ## Rounding bin edges

    ##Properly padding out C so the colors go with the right sectors
    #start[0] = time.time()
    C = zeros((rangeBins.size, TH.size))
    oldfill = 0
    TH_ = TH.tolist()

    for i in range(theta.size):
        fillto = TH_.index(theta[i])
        for j, x in enumerate(colorMap[:,i]):
            C[j, oldfill:fillto].fill(x)
        oldfill = fillto

    #axPolar.clear() ## Clearing plot before writing new data

    plt.title(filename[-27:-4])
    info = 'Roll: '+str(roll)[0:4]+'\nPitch: '+str(pitch)[0:4]+'\nHeading: '+str(heading)[0:4]
    plt.text(0.85, 1.02, info,
      horizontalalignment='left',
      verticalalignment='top',
      size='large',
      bbox=dict(facecolor='white', alpha=1.0),
      transform=plt.gca().transAxes)

    ## Polar plot setup ##
    axPolar.set_theta_direction(1) #Rotation plotting direction
    axPolar.set_theta_zero_location('N', offset=360-157.5) #Zero location north instead of east. Needs to be set according to PCB mounted orientation!
    #axPolar.set_theta_offset(np.deg2rad(heading)) #Rotating plot with compass heading

    northArrow = np.full((10,), np.deg2rad(heading+157.5))
    southArrow = np.full((10,), np.deg2rad(heading+157.5+180))
    r = np.arange(rangeBins[-1]-10, rangeBins[-1])
    axPolar.plot(northArrow, r, color='red')
    axPolar.plot(southArrow, r, color='white')

    ## Setting range and theta ticks/labels ##
    #axPolar.set_xticks(np.arange(0,2.0*np.pi,np.pi/4.0))
    #axPolar.set_xticklabels(['S', 'SE', 'E', 'NE', 'N', 'NW', 'W', 'SW'])
    axPolar.set_yticks(rangeTicks)
    axPolar.set_yticklabels(rangeLabels) #Range labels in meters
    #axPolar.tick_params(colors='red')

    ## Plotting meshgrid ##
    th, r = meshgrid(TH, rangeBins)

    axPolar.pcolormesh(th, r, C, cmap='cividis', shading='gouraud', vmin=0, vmax=1)# shading='gouraud' gives smoothing
    axPolar.grid()


    ## Normalizing detector output from 0 to 1 ##
    CH1_Det = normalizeData(CH1_Det)
    CH2_Det = normalizeData(CH2_Det)


    CH1_Det_idx = np.asarray(np.where(CH1_Det > 0.0)) ## To only plot actual detections
    CH2_Det_idx = np.asarray(np.where(CH2_Det > 0.0)) ## To only plot actual detections

    thetaArr_1 = np.full((CH1_Det_idx.shape[1]), ((sector+1)*2 - 1)*np.pi/8)
    thetaArr_2 = np.full((CH2_Det_idx.shape[1]), (sector*2 - 1)*np.pi/8)

    ## Plotting normalized detector output in corresponding sector ##
    axPolar.scatter(thetaArr_1, rangeBins[CH1_Det_idx], c=CH1_Det[CH1_Det_idx], cmap='RdPu_r', vmin=0, vmax=1) ## Plotting CH1 detections, colormapped
    axPolar.scatter(thetaArr_2, rangeBins[CH2_Det_idx], c=CH2_Det[CH2_Det_idx], cmap='RdPu_r', vmin=0, vmax=1) ## Plotting CH2 detections, colormapped
    plt.draw()


    plt.pause(1e-5) ## This is needed due to some weird stuff with plot updates

def RX_polarPlot(CH1_Intensity, CH2_Intensity, zone, heading, CH1_Det, CH2_Det, tilt, sectorFocus=False):
    ''' Plots the received echo intensities on a polar plot
    '''
    global rangeLabels, rangeTicks
    sector = zone*2-1
    ''' Assigning colormap to sampled sector for plotting '''

    if sectorFocus:
        for rangeVal in range(0, nBins):
            colorMap[rangeVal, sector+1] = CH1_Intensity[rangeVal]
    else:
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

    ## Polar plot setup ##
    print("\n Heading:", heading)
    print("\n")
    axPolar.set_theta_direction(1) #Rotation plotting direction
    axPolar.set_theta_zero_location('N', offset=360-157.5) #Zero location north instead of east. Needs to be set according to PCB mounted orientation!
    #axPolar.set_theta_offset(np.deg2rad(heading)) #Rotating plot with compass heading

    #northArrow = np.full((20,), np.deg2rad(heading+157.5))
    northArrow = np.full((50,), np.deg2rad(heading+157.5))
    eastArrow = np.full((50,), np.deg2rad(heading+157.5+90))
    westArrow = np.full((50,), np.deg2rad(heading+157.5-90))
    southArrow = np.full((50,), np.deg2rad(heading+157.5+180))

    r = np.arange(rangeBins[-1]-50, rangeBins[-1])
    axPolar.plot(northArrow, r, color='red')
    axPolar.plot(eastArrow, r, color='white')
    axPolar.plot(southArrow, r, color='white')
    axPolar.plot(westArrow, r, color='white')

    #axPolar.arrow(np.deg2rad(heading+157.5), int(len(CH1_Intensity)*(2/4)), 0,
    #              int(len(CH1_Intensity)*(1/8)), width=0.05, alpha = 1, linewidth=2,
    #              edgecolor = 'red', facecolor = 'red')


    #northArrow = np.full((10,), np.deg2rad(heading+157.5))
    #southArrow = np.full((10,), np.deg2rad(heading+157.5+180))
    #r = np.arange(rangeBins[-1]-10, rangeBins[-1])
    #axPolar.plot(northArrow, r, color='red')
    #axPolar.plot(southArrow, r, color='white')

    ## Setting range and theta ticks/labels ##
    #axPolar.set_xticks(np.arange(0,2.0*np.pi,np.pi/4.0))
    #axPolar.set_xticklabels(['S', 'SE', 'E', 'NE', 'N', 'NW', 'W', 'SW'])
    axPolar.set_yticks(rangeTicks)
    axPolar.set_yticklabels(rangeLabels) #Range labels in meters
    axPolar.tick_params(colors='red')

    ## Tilt info text:
    #info = 'Roll: '+str(roll)[0:4]+'\nPitch: '+str(pitch)[0:4]+'\nHeading: '+str(heading)[0:4]
    '''
    info = 'Tilt: '+str(tilt)+' from vertical axis\nHeading: '+str(heading)[0:4]

    plt.text(-0.2, 1.02, info,
      horizontalalignment='left',
      verticalalignment='top',
      size='large',
      bbox=dict(facecolor='red', alpha=1.0),
      transform=plt.gca().transAxes)
    '''


    ## Plotting meshgrid ##
    th, r = np.meshgrid(TH, rangeBins)
    axPolar.pcolormesh(th, r, C, cmap='cividis', shading='gouraud', vmin=0, vmax=1)# shading='gouraud' gives smoothing
    axPolar.grid()

    ## Normalizing detector output from 0 to 1 ##
    CH1_Det = normalizeData(CH1_Det)
    CH2_Det = normalizeData(CH2_Det)

    CH1_Det_idx = np.asarray(np.where(CH1_Det > 0.0)) ## To only plot actual detections
    CH2_Det_idx = np.asarray(np.where(CH2_Det > 0.0)) ## To only plot actual detections

    thetaArr_1 = np.full((CH1_Det_idx.shape[1]), (sector*2 - 1)*np.pi/8)
    thetaArr_2 = np.full((CH2_Det_idx.shape[1]), ((sector+1)*2 - 1)*np.pi/8)

    ## Plotting normalized detector output in corresponding sector ##
    axPolar.scatter(thetaArr_1, rangeBins[CH1_Det_idx], c=CH1_Det[CH1_Det_idx], cmap='RdPu_r', vmin=0, vmax=1) ## Plotting CH1 detections, colormapped
    axPolar.scatter(thetaArr_2, rangeBins[CH2_Det_idx], c=CH2_Det[CH2_Det_idx], cmap='RdPu_r', vmin=0, vmax=1) ## Plotting CH2 detections, colormapped
    plt.draw()

    plt.pause(1e-5) ## This is needed due to some weird stuff with plot updates

def RX_polarPlot_SectorFocus(CH1_Intensity, CH2_Intensity, zone, heading, CH1_Det, CH2_Det, tilt):
    ''' Plots the received echo intensities on a polar plot
    '''
    global rangeLabels, rangeTicks
    sector = zone*2-1
    ''' Assigning colormap to sampled sector for plotting '''
    for rangeVal in range(0, nBins):
        colorMap[rangeVal, sector] = CH1_Intensity[rangeVal]
        colorMap[rangeVal, sector+1] = CH2_Intensity[rangeVal]


    TH = cbook.simple_linear_interpolation(theta, 5) ## Rounding bin edges

    ##Properly padding out C so the colors go with the right sectors
    #start[0] = time.time()
    C = zeros((rangeBins.size, TH.size))
    oldfill = 0
    TH_ = TH.tolist()
    for i in range(theta.size):
        fillto = TH_.index(theta[i])
        for j, x in enumerate(colorMap[:,i]):
            C[j, oldfill:fillto].fill(x)
        oldfill = fillto

    axPolar.clear() ## Clearing plot before writing new data

    ## Polar plot setup ##
    axPolar2.set_theta_direction(1) #Rotation plotting direction
    axPolar2.set_theta_zero_location('N', offset=360-157.5) #Zero location north instead of east. Needs to be set according to PCB mounted orientation!
    #axPolar.set_theta_offset(np.deg2rad(heading)) #Rotating plot with compass heading

    northArrow = np.full((len(CH1_Intensity)*0.2,), np.deg2rad(heading+157.5))
    eastArrow = np.full((len(CH1_Intensity)*0.1,), np.deg2rad(heading+157.5+90))
    westArrow = np.full((len(CH1_Intensity)*0.1,), np.deg2rad(heading+157.5-90))
    southArrow = np.full((10,), np.deg2rad(heading+157.5+180))
    r = np.arange(rangeBins[-1]-10, rangeBins[-1])
    axPolar2.plot(northArrow, r, color='red')
    axPolar2.plot(eastArrow, r, color='white')
    axPolar2.plot(southArrow, r, color='white')
    axPolar2.plot(westArrow, r, color='white')

    ## Setting range and theta ticks/labels ##

    axPolar2.set_yticks(rangeTicks)
    axPolar2.set_yticklabels(rangeLabels) #Range labels in meters
    axPolar2.tick_params(colors='red')


    ## Plotting meshgrid ##
    th, r = meshgrid(TH, rangeBins)
    axPolar2.pcolormesh(th, r, C, cmap='cividis', shading='gouraud', vmin=0, vmax=1)# shading='gouraud' gives smoothing
    axPolar2.grid()

    ## Normalizing detector output from 0 to 1 ##
    CH1_Det = normalizeData(CH1_Det)
    CH2_Det = normalizeData(CH2_Det)
    #print("Normalized CH1 data:", CH1_Det)
    #print("Normalized CH2 data:", CH2_Det)

    CH1_Det_idx = np.asarray(np.where(CH1_Det > 0.0)) ## To only plot actual detections
    CH2_Det_idx = np.asarray(np.where(CH2_Det > 0.0)) ## To only plot actual detections

    #thetaArr_1 = np.full((CH1_Det_idx.shape[1]), ((sector+1)*2 - 1)*np.pi/8)
    #thetaArr_2 = np.full((CH2_Det_idx.shape[1]), (sector*2 - 1)*np.pi/8)

    thetaArr_1 = np.full((CH1_Det_idx.shape[1]), (sector*2 - 1)*np.pi/8)
    thetaArr_2 = np.full((CH2_Det_idx.shape[1]), ((sector+1)*2 - 1)*np.pi/8)

    ## Plotting normalized detector output in corresponding sector ##
    axPolar2.scatter(thetaArr_1, rangeBins[CH1_Det_idx], c=CH1_Det[CH1_Det_idx], cmap='RdPu_r', vmin=0, vmax=1) ## Plotting CH1 detections, colormapped
    axPolar2.scatter(thetaArr_2, rangeBins[CH2_Det_idx], c=CH2_Det[CH2_Det_idx], cmap='RdPu_r', vmin=0, vmax=1) ## Plotting CH2 detections, colormapped

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

def matchedFilter(CH1_data, CH2_data, downSampleStep):
    #print("LEN BEFORE:", len(CH1_data))
    CH1_corr = scipy.signal.correlate(CH1_data, mfilt, mode='same', method='fft')
    CH2_corr = scipy.signal.correlate(CH2_data, mfilt, mode='same', method='fft')

    CH1_Env = (abs(scipy.signal.hilbert(CH1_corr)))
    CH2_Env = (abs(scipy.signal.hilbert(CH2_corr)))#20*np.log10

    CH1_Env = CH1_Env[0:len(tVec)]
    CH2_Env = CH2_Env[0:len(tVec)]
    #fig, ax5 = plt.subplots(1)
    #ax5.plot(CH1_Env)
    #plt.show()
    #print("LEN AFTER:", len(CH1_Env))

    CH1_EnvShort = CH1_Env[0:len(CH1_Env):downSampleStep]
    CH2_EnvShort = CH2_Env[0:len(CH2_Env):downSampleStep]


    return CH1_Env, CH2_Env

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
            directory = os.getcwd()+'/Data/SectorFocus/11-03-22'
        else:
            directory = os.getcwd()+'/Data/11-03-22'
    else:
        if sectorFocus:
            directory = os.getcwd()+'/Data/SectorFocus/16-02-22'
        else:
            directory = os.getcwd()+'/Data/16-02-22'

    files = []
    hhmmss_list = []

    #hhmmss = str(re.findall('[0-9]{2}:[0-9]{2}:[0-9]{2}', filename))[2:-2]

    for root, dirs, filenames in os.walk(directory, topdown=False):
        for filename in filenames:
            hhmm = str(re.findall('[0-9]{2}:[0-9]{2}', filename))[2:-2] ## To get HH:MM from filename
            if 'DS' in filename:
                continue

            ## Only add desired files to list
            if startTime <= hhmm <= stopTime and filename.endswith('.npz'):
                files.append(root+'/'+filename)
                #print("Added file:", filename)
                #hhmmss_list.append(hhmmss)

    #hhmmss_list = sort(hhmmss_list)
    #print(files[0][-18:-4])
    #quit()

    ## Sorting files by time
    files = sorted(files, key=lambda x: x[-18:-4])

    #files = sort(files) ## Sorting by time

    return files

def loadVideoFileNames(startTime, stopTime, ace):
    videofiles = []
    hhmmss_list = []
    if args.ace:
        directory = os.getcwd()+'/Data/cam_recordings/secondTest'
    else:
        directory = os.getcwd()+'/Data/cam_recordings/firstTest'

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

    videofiles = sort(videofiles) ## Sorting by time
    #print(videofiles)

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
    args = parser.parse_args()




    if args.syncPlot and not args.o2temp:
        rx_files = loadFileNames(args.startTime, args.stopTime, args.sectorFocus, args.ace)

        videoFiles = loadVideoFileNames(args.startTime, args.stopTime, args.ace)
        for video in videoFiles:
            print("Current video:", video)
            fusePlot.syncPlot_timeStampFromFrames(video, rx_files, sectorFocus=args.sectorFocus, \
                                                savePlots=args.savePlots, showPlots=args.showPlots)

        quit()
    if args.o2temp:
        if args.profile:
            profilePlot.SVProfilePlot(args.ace)
            ## Depth measurements performed in this time window
            files = loadFileNames('09:47', '09:51', args.sectorFocus)
            #for file in files:
            #    print(file)#print(files)
            profilePlot.profilePlot(files)
        else:
            files = loadFileNames(args.startTime, args.stopTime, args.sectorFocus)
            for file in files:
                print(file)#print(files)
            profilePlot.O2TempPlot(files)
    if args.imu:
        files = loadFileNames(args.startTime, args.stopTime, args.sectorFocus)
        IMU.plotData(files)






    quit()
    #fig3, ax3 = plt.subplots(1)

    #fig, (ax1, ax2) = plt.subplots(2, figsize=(9,6))

    fig, ax = plt.subplots(1)

    for filename in rx_files:
        #fig, (ax1, ax2) = plt.subplots(2, figsize=(9,6))

        data=np.load(filename, allow_pickle=True)

        print("Current file:", filename[-46:-4])


        if len(data['header']) > 5:
            ''' Since header-contents were changed at some point '''
            acqInfo = data['header']
            imuData = data['IMU']
            O2Data = data['O2']
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



        c = 1463
        downSampleStep = 1




        #tfilt = np.linspace(0, pulseLength, 1000)
        #mfilt = scipy.signal.chirp(tfilt, fc-BW/2, pulseLength, fc+BW/2, method='linear')
        #plt.plot(mfilt)
        #plt.show()






        ## Differentiate full sectorscan from only sector 4 scan
        ## will be fixed by filename etc later

        if data['sectorData'].ndim == 1:
            Sector4_data = data['sectorData'][:]
            nSamples = len(Sector4_data)
            sectorFocus = True
        else:
            #roll = imuData[0][0]
            #pitch = imuData[1]
            #heading = imuData[2]


            #print("IMU data:", imuData[0][0:3])
            #print("IMU data:", imuData[1][0:3])
            #print("IMU data:", imuData[2][0:3])
            #print("IMU data:", imuData[3][0:3])
            Sector1_data = data['sectorData'][:,0]
            Sector2_data = data['sectorData'][:,1]
            Sector3_data = data['sectorData'][:,2]
            Sector4_data = data['sectorData'][:,3]
            Sector5_data = data['sectorData'][:,4]
            Sector6_data = data['sectorData'][:,5]
            Sector7_data = data['sectorData'][:,6]
            Sector8_data = data['sectorData'][:,7]
            nSamples = len(Sector1_data)


        ### Acquisition constants ###
        #SampleTime = Range*2.0/c # How long should we sample for to cover range
        SampleTime = nSamples*(1/fs)
        Range = c*SampleTime/2
        #nSamples = int(fs*SampleTime) # Number og samples to acquire per ping
        samplesPerPulse = int(fs*pulseLength)  # How many samples do we get per pulse length
        tVec = np.linspace(0, SampleTime, nSamples)
        tVecShort = tVec[0:len(tVec):downSampleStep] # Downsampled time vector for plotting
        rangeVec = np.linspace(0, Range, len(tVec))
        rangeVecShort = np.linspace(0, Range, len(tVecShort)).round(decimals=2)


        ## Matched filter
        #tfilt = np.linspace(0, pulseLength, int(fs*pulseLength))
        #mfilt = scipy.signal.chirp(tfilt, int(fc-BW/2), tfilt[-1], int(fc+BW/2),method='linear',phi=90)


        ## Hamming window on matched filter sig ##
        #mfilt = mfilt*np.hamming(len(mfilt))*1.85

        mfilt = acousticProcessing.gen_mfilt(fc, BW, pulseLength, fs)



        #fig2, ax2 = plt.subplots(2,figsize=(7,6))
        #ax3 = ax2.twinx()  # instantiate a second axes that shares the same x-axis
        #move_figure(fig2, 600, 0)

        if args.sectorFocus:

            roll = imuData[0]
            pitch = imuData[1]
            heading = imuData[2]

            a = np.arctan(np.deg2rad(roll))*np.arctan(np.deg2rad(roll))
            b = np.arctan(np.deg2rad(pitch))*np.arctan(np.deg2rad(pitch))
            tilt = np.degrees(np.arctan(sqrt(a+b)))
            CH1_Env, _ = acousticProcessing.matchedFilter(Sector4_data, Sector4_data, mfilt, downSampleStep)
            CH1_peaks_idx, CH1_noise, CH1_detections, CH1_thresholdArr = acousticProcessing.peakDetect(CH1_Env, num_train=6, num_guard=2, rate_fa=5e-3)
            CH1_Intensity, _ = acousticProcessing.colorMapping(CH1_Env, _)


            #CH1_Env, _ = matchedFilter(Sector4_data, Sector4_data, downSampleStep)

            #CH1_peaks_idx, CH1_noise, CH1_detections, CH1_thresholdArr = peakDetect(CH1_Env, num_train=6, num_guard=2, rate_fa=5e-3)

            #CH1_Intensity, _ = colorMapping(CH1_Env, _)
            #CH1_Env[0:samplesPerPulse] = 0.00001
            ax.clear()

            #plt.subplots(211)
            #ax2[0].plot(rangeVec, CH1_Samples, label='Signal from '+channelArray[2*(zone-1)][0])
            ax.plot(rangeVecShort, CH1_Env, label='Signal from Sector 4')
            #ax2[0].plot(freqs, CH1_fft, label='Signal from '+channelArray[2*(zone-1)][0])
            ax.legend()

            plt.draw()
            plt.pause(1e-6)
            plt.waitforbuttonpress()

            continue
        #polarPlot_init(tVecShort, rangeVecShort)
        else:
            for zone in range(1,5):
                roll = imuData[zone-1][0]
                pitch = imuData[zone-1][1]
                heading = imuData[zone-1][2]

                a = np.arctan(np.deg2rad(roll))*np.arctan(np.deg2rad(roll))
                b = np.arctan(np.deg2rad(pitch))*np.arctan(np.deg2rad(pitch))
                tilt = np.degrees(np.arctan(sqrt(a+b)))

                CH1_Samples, CH2_Samples = data['sectorData'][:,zone*2-1], data['sectorData'][:,2*(zone-1)]

                CH1_Env, CH2_Env = matchedFilter(CH1_Samples, CH2_Samples, downSampleStep)
                #CH1_Env, CH2_Env, CH1_Freq, CH2_Freq = Hilbert(CH1_Samples, CH2_Samples, downSampleStep)
                CH1_peaks_idx, CH1_noise, CH1_detections, CH1_thresholdArr = peakDetect(CH1_Env, num_train=6, num_guard=2, rate_fa=5e-3)
                CH2_peaks_idx, CH2_noise, CH2_detections, CH2_thresholdArr = peakDetect(CH2_Env, num_train=6, num_guard=2, rate_fa=5e-3)

                CH1_Intensity, CH2_Intensity = colorMapping(CH1_Env, CH2_Env)


                RX_polarPlot(CH1_Intensity, CH2_Intensity, zone, heading, CH1_detections, CH2_detections, tilt)

                ax2[0].clear()
                ax2[1].clear()
                #plt.subplots(211)
                #ax2[0].plot(rangeVec, CH1_Samples, label='Signal from '+channelArray[2*(zone-1)][0])
                ax2[0].plot(rangeVecShort, CH1_Env, label='Signal from '+channelArray[2*(zone-1)][0])
                #ax2[0].plot(freqs, CH1_fft, label='Signal from '+channelArray[2*(zone-1)][0])

                #plt.subplots(212)
                #ax2[1].plot(rangeVec, CH2_Samples, label='Signal from '+channelArray[zone*2-1][0])
                ax2[1].plot(rangeVecShort, CH2_Env, label='Signal from '+channelArray[zone*2-1][0])
                #ax2[1].plot(freqs, CH2_fft, label='Signal from '+channelArray[2*(zone-1)][0])

                #ax3.plot(tVecShort[CH2_peaks_idx], CH2_detections[CH2_peaks_idx], 'rD')
                ax2[0].legend()
                ax2[1].legend()

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

        plt.tight_layout()

        ax1.legend()
        ax2.legend()
        #ax3.legend()

        #lastData = sig
        plt.show()




        #CH1_EnvShort, CH2_EnvShort, CH1_Freq, CH2_Freq = Hilbert(Sector1_data, sig, downSampleStep)
        #plt.plot(CH2_Freq)
        #plt.show()
        #ax2.plot(CH1_Freq)





        #ax2.plot(corrsig, color='black', label='Matched filter output from raw sig')
        #ax2.plot(sigEnv, color='red', label='Envelope from Hilbert')

        #ax3.plot(noisysig, color='red', label="Raw samples + noise")

        #ax4.plot(noisycorrsig, color='black', label='Matched filter output from noisy sig')
        #ax4.plot(noisysigEnv, color='red', label='Envelope from Hilbert')
        ##ax2.plot(rawcorrsig, label='Matched filter output from raw sig', alpha=0.7)
        ##ax2.plot(Sector1_data, label="Sector data")

        #ax2.plot(tVec, Sector4_data)
        #plt.title(fs)

        #ax3.legend()

    #plt.show()
