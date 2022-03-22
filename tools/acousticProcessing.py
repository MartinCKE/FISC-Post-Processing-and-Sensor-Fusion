import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfilt, sosfreqz, filtfilt


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

    VGA_Gres = 50 # 50dB per volt
    ## LO gain mode: G = 50 dB/V * Vgain - 6.50dB
    ## HI gain mode: G = 50 dB/V * Vgain + 5.5
        # VgainLO = (G+6.5) / 50
        # VgainHI = (G-5.5) / 50

    #c = 1500 ## Sound speed in water
    resolution = 1.214/1024 ## 10 bit DAC Vrange = 0-1.214V

    #times = np.arange(0, Range*2.0/c, 1/fs, dtype=np.float64)
    #times = np.arange(0, Range/c, 1/fs, dtype=np.float64)
    times = np.linspace(0,Range/c, len(data))

    ranges = c*times
    gainArr = np.zeros((len(ranges)))

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

        #alpha = 1.2# ## Custom threshold (rate_fa ignored)
        threshold = alpha * p_noise
        #print(threshold)
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

    print("alpha", alpha)

    noiseArr.append(p_noise)
    peak_idx = np.array(peak_idx, dtype=int)

    detectorarr = np.log10(data/peaks)

    #fig, ax = plt.subplots(1)
    #ax.plot(data, color='red', label='data', alpha=0.5)
    #ax.plot(noiseArr, color='black', label='noise', alpha=0.5)
    #ax.plot(peaks, color='magenta', label='peaks', alpha=0.5)
    #ax.plot(thresholdArr, color='blue', label='threshold', alpha=0.5)
    #ax.plot(detectorarr, color='black', label='detectorarr')
    #plt.legend()
    #plt.show()
    #quit()


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

def matchedFilter(CH1_data, CH2_data, mfilt, downSampleStep):
    #print("LEN BEFORE:", len(CH1_data))
    CH1_corr = scipy.signal.correlate(CH1_data, mfilt, mode='same', method='fft')
    CH2_corr = scipy.signal.correlate(CH2_data, mfilt, mode='same', method='fft')

    CH1_Env = (abs(scipy.signal.hilbert(CH1_corr)))
    CH2_Env = (abs(scipy.signal.hilbert(CH2_corr)))#20*np.log10

    CH1_EnvShort = CH1_Env[0:len(CH1_Env):downSampleStep]
    CH2_EnvShort = CH2_Env[0:len(CH2_Env):downSampleStep]


    return CH1_Env, CH2_Env
