import numpy as np
import matplotlib.pyplot as plt
import os
import re
from scipy.signal import find_peaks
import datetime
import argparse
import tools.acousticProcessing

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
    rangeVec = np.linspace(0, Range, len(tVec))
    rangeVecShort = np.linspace(0, Range, len(tVecShort)).round(decimals=2)
    rxdata[0:samplesPerPulse] = 0

    mfilt = acousticProcessing.gen_mfilt(fc, BW, pulseLength, fs)
    rxdata, _ = acousticProcessing.matchedFilter(rxdata, rxdata, mfilt, downSampleStep)

    d=1 ## some range shit is wrong with FISC
    range_idx = round((2*d*fs)/c)
    min_idx = range_idx# - 600
    max_idx = range_idx + 200
    #plt.plot(rangeVec[min_idx:max_idx], rxdata[min_idx:max_idx])
    #plt.plot(rangeVec,rxdata, label=file[-20::])
    #plt.legend()
    #plt.draw()
    #plt.pause(0.2)

    #print(file)
    #plt.plot(rangeVec, rxdata)
    #plt.show()
    return rxdata[min_idx:max_idx]

def calcbeam(files, Shading):
    beam = []
    angles = []
    startTime = datetime.datetime.strptime(files[0][-18:-4], '%H:%M:%S.%f')
    endTime = datetime.datetime.strptime(files[-1][-18:-4], '%H:%M:%S.%f')
    t = endTime-startTime
    t = t.total_seconds()  ##Time from depth= 0 to 2.3m
    ## TESTING ##
    #t = 10

    depths = np.linspace(0, 2.3, len(files)) ## since ball was lowered with constant velocity


    for i, file in enumerate(files):
        data = getEchoData(file) ## matched filter output
        #d = infoarr[i][0]

        angle = np.arctan((1.15-depths[i]))
        print("angle:", angle)
        #peaks, _ = find_peaks(data, height=0)
        #data = acousticProcessing.normalizeData(data)
        data = 20*np.log10(data)

        #plt.plot(data)
        #plt.show()
        beam.append(np.max(data))
        #plt.plot(data, label=file[-20::])
        #plt.show()


        angles.append(angle)#+np.pi/2)
        #plt.plot(peaks, data[peaks], "x")
        #plt.legend()
        #plt.show()
    beam -= np.max(beam) ## normalizing to 0 dB

    #data = acousticProcessing.normalizeData(data)
    #plt.plot(data, color='red')
    #plt.show()
    #quit()

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.set_theta_zero_location("N")

    ax.set_thetamax(90)
    ax.set_thetamin(-90)
    if Shading:
        ax.set_title('FISC Vertical Beam Pattern, RX_h = 10mm')
    else:
        ax.set_title('FISC Vertical Beam Pattern, RX_h = 20mm (original)')
    ax.plot(angles, beam)
    #plt.tight_layout()

    plt.show()




def loadFileNames(startTime, endTime, Shading):#, arr):

    directory = os.getcwd()+'/Data/SectorFocus/10-03-22'
    files = []
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
                files.append(root+'/'+filename)

    ## Sorting files by time
    files = sorted(files, key=lambda x: x[-18:-4])
    return files

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--BW",
        action="store",
        type=int,
        default='00:00',
        dest='BW',
        help="BW, 0 or 20e3",
    )
    parser.add_argument(
        "--shading",
        type=int,
        action="store",
        dest="shading",
        help="Shading in mm (0, 10 or 15)",
    )
    parser.add_argument(
        "--sector",
        type=int,
        action="store",
        dest="sector",
        help="Since tests were performed on sector 3 as well to compare",
    )
    args = parser.parse_args()


    if args.BW == 0 and args.shading == None:
        Shading = False
        startTime = '13:15:00' ##0kHz BW run 2
        endTime = '13:15:20'

    elif args.BW == 20 and args.shading == None:
        Shading = False
        startTime = '12:07:58' ##20kHz BW
        endTime = '12:08:15'

    elif args.BW == 0 and args.shading == 10:
        startTime = '13:19:12'
        endTime =  '13:19:27'
        Shading=True

    elif args.BW == 0 and args.shading == 10:
        pass

    elif args.BW == 40 and args.shading == 10:
        startTime = '14:57:15'
        endTime =  '14:57:31'
        Shading=True
    elif args.BW == 40 and args.shading == None:
        startTime = '15:00:27'
        endTime =  '15:00:48'
        Shading=False
    elif args.BW == 40 and args.shading == 15:
        startTime = '15:06:19'
        endTime =  '15:06:41'
        Shading=True

    #startTime = '18:34:11' ## Sector 3 10mm shading
    #endTime = '18:34:44'
        ## 10mm shading  14:57:15.3 - 14:57:31.8
        ## uten shading_gradient  15:00:27.86 - 15:00:46.91
        ## 15 mm shading 15:06:19.0 - 15:06:41.76

    '''
    Shading = False
    if Shading:
        arr = Shading_10mm_1m
        #startTime = '11:56:06' ## 11:55:20
        #endTime = '11:56:29'
        ### 0kHz BW
        startTime = '11:57:38'
        endTime =  '11:58:04'
        ###
        ## 0kHz run 2
        #startTime = '13:19:12'
        #endTime =  '13:19:27'
    else:
        arr = noShading_1m
        startTime = '12:07:58' ##20kHz BW
        endTime = '12:08:15'   ##20kHz BW
        #startTime = '13:15:00' ##0kHz BW run 2
        #endTime = '13:15:20'
    '''
    filenames = loadFileNames(startTime, endTime, Shading)#, arr)

    calcbeam(filenames, Shading)
    #print(filenames)

if __name__ == '__main__':
    main()
