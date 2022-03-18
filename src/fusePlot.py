# Import required packages
import cv2
import pytesseract
import tesserocr
import time
from PIL import Image
import numpy as np
import re
import datetime
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
import os

matplotlib.use('TkAgg')

#Import from other scripts
from tools.acousticProcessing import gen_mfilt, matchedFilter, colorMapping, TVG, peakDetect, normalizeData

def get_concat_h_resize(im1, im2, resample=Image.BICUBIC, resize_big_image=True):
	if im1.height == im2.height:
		_im1 = im1
		_im2 = im2
	elif (((im1.height > im2.height) and resize_big_image) or
		  ((im1.height < im2.height) and not resize_big_image)):
		_im1 = im1.resize((int(im1.width * im2.height / im1.height), im2.height), resample=resample)
		_im2 = im2
	else:
		_im1 = im1
		_im2 = im2.resize((int(im2.width * im1.height / im2.height), im1.height), resample=resample)
	dst = Image.new('RGB', (_im1.width + _im2.width, _im1.height))
	dst.paste(_im1, (0, 0))
	dst.paste(_im2, (_im1.width, 0))
	return dst

def get_concat_v_resize(im1, im2, resample=Image.BICUBIC, resize_big_image=True):
	if im1.width == im2.width:
		_im1 = im1
		_im2 = im2
	elif (((im1.width > im2.width) and resize_big_image) or
		  ((im1.width < im2.width) and not resize_big_image)):
		_im1 = im1.resize((im2.width, int(im1.height * im2.width / im1.width)), resample=resample)
		_im2 = im2
	else:
		_im1 = im1
		_im2 = im2.resize((im1.width, int(im2.height * im1.width / im2.width)), resample=resample)
	dst = Image.new('RGB', (_im1.width, _im1.height + _im2.height))
	dst.paste(_im1, (0, 0))
	dst.paste(_im2, (0, _im1.height))
	return dst

def move_figure(f, x, y):
	'''Move figure's upper left corner to pixel (x, y)'''
	backend = matplotlib.get_backend()
	if backend == 'TkAgg':
		f.canvas.manager.window.wm_geometry("+%d+%d" % (x, y))
	elif backend == 'WXAgg':
		f.canvas.manager.window.SetPosition((x, y))
	else:
		f.canvas.manager.window.move(x, y)

def inclinationHeading(imuData):
	roll = imuData[0]
	pitch = imuData[1]
	heading = imuData[2]

def parseVideoTime(filename):

	try:
		YYMMDDhhmmssff = str(re.findall('[0-9]{4}-[0-9]{2}-[0-9]{2}-[0-9]{2}-[0-9]{2}-[0-9]{2}.[0-9]{5}', filename))[2:-2]
		print("kbhk", YYMMDDhhmmssff)
		startTime = datetime.datetime.strptime(YYMMDDhhmmssff, '%Y-%m-%d-%H-%M-%S.%f')

		endTime = startTime + datetime.timedelta(seconds=15)
		startTime = str(startTime)[-15:]# + '.0'
		endTime = str(endTime)[-15:]# +'.0'

	except:
		YYMMDDhhmmss = str(re.findall('[0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{2}-[0-9]{2}-[0-9]{2}', filename))[2:-2]
		startTime = datetime.datetime.strptime(YYMMDDhhmmss, '%Y-%m-%d_%H-%M-%S')
		endTime = startTime + datetime.timedelta(seconds=15)

		startTime = str(startTime)[-8:]# + '.0'
		endTime = str(endTime)[-8:]# +'.0'

	return startTime, endTime

def genSyncPlot(axs, frame, video_timeStamp, rx_file, rxFile_timeStamp, **kwargs):
	''' Generates plot of RX data with a video frame from same time (ish).
		Ish since the video timestamp doesn't have milliseconds accuracy, so estimate
		timestamp based on framerate and current frame.
	'''


	data=np.load(rx_file, allow_pickle=True)
	acqInfo = data['header']
	imuData = data['IMU']
	#O2Data = data['O2']
	fc = int(acqInfo[0])
	BW = int(acqInfo[1])
	pulseLength = acqInfo[2]
	fs = acqInfo[3]
	#Range = int(acqInfo[4]) Range is re-computed due to sound velocity mismatch during acquisition
	#c = acqInfo[5]
	downSampleStep = int(acqInfo[6])


	#inclination, currentDir = inclinationHeading(imuData)

	c = 1463 ## Based on profiler data
	downSampleStep = 1

	if data['sectorData'].ndim == 1 or kwargs.get("sectorFocus"):
		Sector4_data = data['sectorData'][:]
		nSamples = len(Sector4_data)
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


	nSamples = len(Sector4_data)
	#sectorFocus = True

	### Acquisition constants ###
	SampleTime = nSamples*(1/fs)
	Range = c*SampleTime/2
	samplesPerPulse = int(fs*pulseLength)  # How many samples do we get per pulse length
	tVec = np.linspace(0, SampleTime, nSamples)
	tVecShort = tVec[0:len(tVec):downSampleStep] # Downsampled time vector for plotting
	rangeVec = np.linspace(0, Range, len(tVec))
	rangeVecShort = np.linspace(0, Range, len(tVecShort)).round(decimals=2)

	#print("fc:", int(fc), "BW:", int(BW), "fs:", int(fs), \
	#	"plen (us):", int(pulseLength*1e6), "range:", Range, "c:", c, "Downsample step:", downSampleStep)
	#print("Sample Time:", SampleTime, "Nsamples:", nSamples)



	mfilt = gen_mfilt(fc, BW, pulseLength, fs)


	'''
	#import scipy
	#tfilt = np.linspace(0, pulseLength, int(fs*pulseLength))
	mfilt2 = acousticProcessing.gen_mfilt(fc, BW, pulseLength, fs)#scipy.signal.chirp(tfilt, int(fc-BW/2), tfilt[-1], int(fc+BW/2),method='linear',phi=90)
	#mfilt2 = mfilt2*np.hamming(len(mfilt2))*1.85
	CH1_Env2, _ = acousticProcessing.matchedFilter(Sector4_data, Sector4_data, mfilt2, downSampleStep)
	#CH1_Env2[0:samplesPerPulse] = 0
	#axs.plot(rangeVecShort, CH1_Env2, label=rxFile_timeStamp, color='black', alpha=0.5)

	#axs.plot(rangeVecShort, Sector4_data, label=rxFile_timeStamp, color='red', alpha=0.5)
	#axs.plot(rangeVecShort, Sector4_data, label='gained', color='black', alpha=0.5)
	fig, ax = plt.subplots(1)
	#freqs=np.linspace(0,fs*2, len(mfilt))
	#ax.plot(freqs, 20*np.log10(np.fft.fft(Sector4_data[0:samplesPerPulse])))
	#ax.plot(freqs, 20*np.log10(np.fft.fft(mfilt2)), color='red')
	quit()
	'''



	'''
	filtered = acousticProcessing.butterworth_BP_filter(Sector4_data, 98802, 158802, fs, 10)
	mfilt = Sector4_data[0:samplesPerPulse]
	mfiltfft = np.fft.rfft(mfilt)

	n = mfiltfft.size
	freqs = np.fft.rfftfreq(n*2-1, 1/fs)
	#freqs2=np.linspace(0,fs/2, len(mfiltfft))
	plt.plot(freqs, np.abs(mfiltfft), alpha=0.5)
	#plt.plot(freqs2, np.abs(mfiltfft), color='black', alpha=0.5)
	plt.show()
	quit()
	'''

	CH1_Env, _ = matchedFilter(Sector4_data, Sector4_data, mfilt, downSampleStep)
	CH1_Env_TVG = TVG(CH1_Env, Range, c, fs)
	CH1_Env[0:samplesPerPulse] = 0 ## To remove tx pulse noise
	CH1_Env_TVG[0:samplesPerPulse] = 0 ## To remove tx pulse noise
	CH1_Env = normalizeData(CH1_Env)
	#axs.plot(rangeVecShort, CH1_Env_TVG, label=rxFile_timeStamp, color='red', alpha=0.5)
	#axs.plot(rangeVecShort, CH1_Env, label=rxFile_timeStamp, color='red', alpha=0.5)
	#print(len(CH1_Env))
	#quit()
	CH1_peaks_idx, CH1_noise, CH1_detections, CH1_thresholdArr = peakDetect(CH1_Env, num_train=50, num_guard=10, rate_fa=0.3)
	print(CH1_peaks_idx)

	#quit()
	CH1_Intensity, _ = colorMapping(CH1_Env, _)

	#CH1_Env[0:samplesPerPulse] = 0.00001
	axs.clear()
	#plt.subplots(211)
	#ax2[0].plot(rangeVec, CH1_Samples, label='Signal from '+channelArray[2*(zone-1)][0])
	#testax =
	axs.plot(rangeVecShort, CH1_Env, label=rxFile_timeStamp+', BW:'+str(BW))
	axs.plot(rangeVecShort[CH1_peaks_idx], CH1_detections[CH1_peaks_idx], 'rD')
	#axs.plot(rangeVecShort, CH1_Env_TVG, label=rxFile_timeStamp, color='black')

	axs.set_title('Replica Correlator output')
	axs.set_xlabel('Range')
	axs.grid(b=True, which="major", color="black", linestyle="-", alpha=0.5)
	#axs.minorticks_on()
	axs.grid(b=True, which="minor", color="#999999", linestyle="--", alpha=0.2)
	#plt.style.use("seaborn-notebook")
	#ax2[0].plot(freqs, CH1_fft, label='Signal from '+channelArray[2*(zone-1)][0])
	axs.legend()
	#plt.tight_layout()
	axs.patch.set_facecolor("#edf3f5")


	x,y,w,h = 5,5,320,32

	# Draw black background rectangle
	cv2.rectangle(frame, (x, x), (x + w, y + h), (0,0,0), -1)

	# Add text
	cv2.putText(frame, 'Video file: '+video_timeStamp, (x + int(w/50),y + int(h/2)+5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
	print("SHOULDKJBKASJBGKASBGABG")
	print("Filename now:", rx_file)
	if kwargs.get('showPlots'):

		plt.draw()
		frame_s = cv2.resize(frame, None, fx=0.7, fy=0.7, interpolation=cv2.INTER_AREA)
		cv2.imshow(video_timeStamp, frame_s)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			quit()

		plt.pause(1)

	if kwargs.get("savePlots"):
		video_timeStamp = video_timeStamp.replace(":", ".")
		rxFile_timeStamp = rxFile_timeStamp.replace(":", ".")
		frame_filename = 'VideoFrame_2022-02-16_'+video_timeStamp+'.png'
		RX_filename = 'RXSample_2022-02-16_'+rxFile_timeStamp+'.png'
		syncPlotName = 'syncPlot_2022-02-16_'+rxFile_timeStamp+'.png'

		cv2.imwrite('temp/'+frame_filename, frame)
		plt.savefig('temp/'+RX_filename, dpi=600)

		im1 = Image.open('temp/'+frame_filename)
		im2 = Image.open('temp/'+RX_filename)

		if kwargs.get("ace"):
			if 'SectorFocus' in rx_file:
				syncPlotName = 'syncPlot_sectorFocus_2022-03-11_'+rxFile_timeStamp+'.png'
		else:
			if 'SectorFocus' in rx_file:
				syncPlotName = 'syncPlot_sectorFocus_2022-02-16_'+rxFile_timeStamp+'.png'

		get_concat_h_resize(im1, im2).save('generatedPlots/'+syncPlotName)
		print(os.getcwd())

		os.remove(os.getcwd()+'/temp/'+frame_filename)
		os.remove(os.getcwd()+'/temp/'+RX_filename)

		#get_concat_v_resize(im1, im2, resize_big_image=False).save('data/dst/pillow_concat_v_resize.jpg')




def syncPlot_timeStampFromFrames(videoFile, all_rx_files, **kwargs):
	''' Parses files and finds video frame which is matched to rx data.
	'''

	startTime, endTime = parseVideoTime(videoFile)
	print("Start time:", startTime, "End time:", endTime)

	rxFilesInVideo = []
	rxFiles_timeStamps = []
	for rx_file in all_rx_files:


		try:
			hhmmssff = str(re.findall('[0-9]{2}:[0-9]{2}:[0-9]{2}.[0-9]{5}', rx_file))[2:-2]
			if startTime <= hhmmssff <= endTime:
				### Only capturing RX files acquired during current video ###
				rxFilesInVideo.append(rx_file)
				rxFile_timeStamp = str(re.findall('[0-9]{2}:[0-9]{2}:[0-9]{2}.?[0-9]{5}', rx_file))[2:-2]
				rxFiles_timeStamps.append(datetime.datetime.strptime(rxFile_timeStamp, '%H:%M:%S.%f'))
				continue

			elif hhmmss > endTime:
				break

		except:
			hhmmss = str(re.findall('[0-9]{2}:[0-9]{2}:[0-9]{2}', rx_file))[2:-2]
			if startTime <= hhmmss <= endTime:
				### Only capturing RX files acquired during current video ###

				rxFilesInVideo.append(rx_file)
				rxFile_timeStamp = str(re.findall('[0-9]{2}:[0-9]{2}:[0-9]{2}.?[0-9]{5}', rx_file))[2:-2]
				rxFiles_timeStamps.append(datetime.datetime.strptime(rxFile_timeStamp, '%H:%M:%S.%f'))
				continue

			elif hhmmss > endTime:
				break

	if len(rxFilesInVideo) == 0:
		return

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

	#calc_timestamps = [0]
	fps = cap.get(cv2.CAP_PROP_FPS)

	#timestamps = [datetime.datetime.strptime(startTime, '%H:%M:%S')]
	timestamps = [startTime]



	i = 0
	fig, axs = plt.subplots(1, figsize=(6,4*1))
	move_figure(fig, 800, 0)
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

			if abs(rxFiles_timeStamps[i] - curr_frame_timestamp) < datetime.timedelta(milliseconds=20):
				#rx_strTimeStamp = datetime.datetime.strptime(rxFiles_timeStamps[i], '%Y-%m-%d_%H-%M-%S')

				rx_strTimeStamp = str(rxFiles_timeStamps[i])[-15:]# +'.0'

				genSyncPlot(axs, frame, startTime, rxFilesInVideo[i], rx_strTimeStamp, **kwargs)
				if i+1 == len(rxFilesInVideo):
					break
				i+=1



			'''
			lastTime = startTime
			counter += 1/fps
			#print(counter)
			continue

			last_frame_timestamp = datetime.datetime.strptime(timestamps[-1], '%H:%M:%S.%f')
			curr_frame_timestamp = last_frame_timestamp + datetime.timedelta(seconds=1/fps)
			print("last:", last_frame_timestamp)
			print("curr:", curr_frame_timestamp)
			#print(str(curr_frame_timestamp)[-12:])
			#timestamps.append(str(curr_frame_timestamp)[-12:])
			#print("timestamps:", timestamps)
			continue

			print(datetime.timedelta(seconds=1/fps))
			print(timestamps[-1] + datetime.timedelta(seconds=1/fps))
			continue
			timestamps.append(timestamps[-1] + datetime.timedelta(seconds=1/fps))
			print(timestamps[-1])
			#print('jhvvuyvuy', cap.get(cv2.CAP_PROP_POS_MSEC)/1000)

			#timestamps.append(timestamps[-1]+ str(cap.get(cv2.CAP_PROP_POS_MSEC)/1000))

			#frame = frame[20:35,568:712] ## Cropped
			#frame = cv2.resize(frame, None, fx=1.3, fy=1.3, interpolation=cv2.INTER_AREA)
			#cv2.imshow('', frame)
			#cv2.waitKey(0)
			#quit()
			#
			'''


	  # Break the loop
		else:
			break

	# Closes all the frames
	cap.release()
	plt.close()
	cv2.destroyAllWindows()




def OCR_fetchTimeStamp(videoFile):
	''' Uses Tesseract OCR (Optical Characater Recoginition) to acquire timestamp from video.
		Sometimes 0's are 6's, and 1's are 7's, so abandoned this method for now.
	'''
	# Create a VideoCapture object
	cap = cv2.VideoCapture(videoFile)

	# Check if file opened successfully
	if (cap.isOpened() == False):
	  print("Unable to read camera feed")

	# Default resolutions of the frame are obtained.The default resolutions are system dependent.
	# We convert the resolutions from float to integer.
	frame_width = int(cap.get(3))
	frame_height = int(cap.get(4))

	# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
	#out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

	while(True):
		ret, frame = cap.read()


		if ret == True:
			frame = frame[20:35,568:712] ## Cropped
			frame = cv2.resize(frame, None, fx=1.3, fy=1.3, interpolation=cv2.INTER_AREA)

			# Convert the image to gray scale
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

			# Performing OTSU threshold
			ret, thresh1 = cv2.threshold(gray, 200, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

			# Specify structure shape and kernel size.
			# Kernel size increases or decreases the area
			# of the rectangle to be detected.
			# A smaller value like (10, 10) will detect
			# each word instead of a sentence.
			rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
			#erosion = cv2.erode(thresh1,rect_kernel,iterations = 1)
			# Applying dilation on the threshold image
			dilation = cv2.dilate(thresh1, rect_kernel, iterations = 1)

			# Finding contours
			contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL,
															cv2.CHAIN_APPROX_NONE)
			#cv2.imshow('',hierarchy)
			#cv2.waitKey(0)
			#quit()
			print("N contours:", len(contours))
			cv2.imshow('',dilation)
			cv2.waitKey(0)
			#quit()
			for cnt in contours:
				x, y, w, h = cv2.boundingRect(cnt)

				#cv2.waitKey(0)
				# Drawing a rectangle on copied image
				frame2 = frame.copy()


				#quit()
				rect = cv2.rectangle(frame2, (x, y), (x + w, y + h), (0, 255, 0), 2)

				# Cropping the text block for giving input to OCR

				cropped = frame2[y:y + h, x:x + w]
				#cv2.imshow('image', cropped)

				# Open the file in append mode
				#file = open("recognized.txt", "a")
				start = time.time()

				# Apply OCR on the cropped image
				text = pytesseract.image_to_string(cropped, config='--psm 6 --oem 3 \
					   -c tessedit_char_whitelist=0123456789-: ') # -c tessedit_char_blacklist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz/{}[]()
				print(text)

				timeis = str(re.findall('[0-9]{2}:[0-9]{2}:[0-9]{2}', text))#[2:-2]
				print("Altered:", timeis)
				#print("Took:", time.time()-start)
			if cv2.waitKey(25) & 0xFF == ord('q'):
				break
		# Write the frame into the file 'output.avi'
		#out.write(frame)

		# Display the resulting frame
		#cv2.imshow('frame',frame)


	  # Break the loop
		else:
			break

	# When everything done, release the video capture and video write objects
	cap.release()
	#out.release()

	# Closes all the frames
	cv2.destroyAllWindows()

if __name__=='__main__':
	OCR_fetchTimeStamp('testvideo.mp4')
