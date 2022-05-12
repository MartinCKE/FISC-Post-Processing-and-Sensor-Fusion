#!/usr/bin/python3

''' Script for visualizing IMU orientation and plotting stored IMU data.
'''


import numpy as np
from vpython import *
import time
import re
import datetime
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os
import subprocess


imuCMD = 'python3 nmea.py /dev/tty.usbmodem11201 RSP'


headingDict = {"N":0, "NNE":22.5, "NE":45, "ENE":67.5, "E":90,\
			   "ESE":112.5, "SE":135, "SSE":157.5, "S":180, \
			   "SSW":202.5, "SW":225, "WSW":247.5, "W":270, \
			   "WNW":292.5, "NW":315, "NNW":337.5}


def init():
	#from nmea import *
	''' Initializes VPython environment for visualizing orientation.
	'''
	global frontArrow, currentArrow, upArrow, vRotationArrow, sideArrow, myObj

	scene.range=5
	scene.background=color.white#hsv_to_rgb(vector(20, 56, 74))#color.blue

	toRad=2*np.pi/360
	toDeg=1/toRad
	scene.forward=vector(-1,-1,-1)

	scene.width=1000
	scene.height=720

	xarrow=arrow(lenght=1, shaftwidth=.1, color=color.red,axis=vector(4,0,0))
	yarrow=arrow(lenght=1, shaftwidth=.1, color=color.green,axis=vector(0,4,0))
	zarrow=arrow(lenght=1, shaftwidth=.1, color=color.blue,axis=vector(0,0,-4))

	frontArrow=arrow(length=3,shaftwidth=.06,color=color.red,axis=vector(1,0,0))
	upArrow=arrow(length=2,shaftwidth=.06,color=color.magenta,axis=vector(0,1,0))
	sideArrow=arrow(length=2,shaftwidth=.06,color=color.orange,axis=vector(0,0,1))

	#kArrow=arrow(length=5,shaftwidth=0.3,color=color.orange,axis=vector(0,0,1))
	#vRotationArrow=arrow(length=5,shaftwidth=0.3,color=color.orange,axis=vector(0,0,1))

	#incVec = arrow(length=3, shaftwidth=0.05, color=color.black, axis=vector(1,0,-1))
	#headingArrow = arrow(length=3, shaftwidth=0.05, color=color.black, axis=vector(1,0,-1))
	currentArrow = arrow(length=5, shaftwidth=0.05, color=color.black, axis=vector(1,0,-1))


	Capsule = cylinder(pos=vector(0,-1,0), axis=vector(0,2,0), radius=.5, opacity=.8, color=color.green)
	bottomLid = sphere(pos=vector(0,-1,0), radius=0.5, opacity=0.8, color=color.green)
	topLid = sphere(pos=vector(0,1,0), radius=0.5, opacity=0.8, color=color.green)
	Cable = cylinder(pos=vector(0,1,0), axis=vector(0,2,0), radius=.1, opacity=.8, color=color.black)
	myObj=compound([Cable, Capsule, bottomLid, topLid])


def quaternionToYawPitchRoll(w, x, y ,z):
	''' Function for converting IMU quaternions to euler angles (Yaw, pith and roll).
		Inputs:
			- [w, x, y, z]: Unit Quaternion
		Output:
			- [roll, pitch, heading] in degrees
	'''

	sqw = w * w
	sqx = x * x
	sqy = y * y
	sqz = z * z

	#X = np.degrees(np.arctan2(2.0*(x*y + z*w), (sqx-sqy-sqz+sqw)))
	#Y = np.degrees(np.arcsin(-2.0*(x*z - y*w) / (sqx+sqy+sqz+sqw)))
	#Z = np.degrees(np.arctan2(2.0*(y*z + x*w), (-sqx-sqy+sqz+sqw)))

	t2 = +2.0 * (w*y - z*x)
	t2 = +1.0 if t2 > +1.0 else t2
	t2 = -1.0 if t2 < -1.0 else t2
	X = np.degrees(np.arcsin(t2)) ## ROLL

	t0 = +2.0 * (w*x + y*z)
	t1 = +1.0 - 2.0 * (x*x + y*y)
	Y = np.degrees(np.arctan2(t0, t1)) ## PITCH



	t3 = +2.0 * (w*z + x*y)
	t4 = +1.0 - 2.0 * (y*y + z*z)
	### Subtracting 90 degrees for easier orientation in capsule
	Z = -np.degrees(np.arctan2(t3, t4))# - 90  ## YAW fjernet "-" foran
	Z+= 4.26 ## Compensation for magnetic declination at Dora location

	if Z < 0:
		Z+=360
	elif Z > 360:
		Z-=360



	return X, Y, Z

def getQuat():
	''' Gets a quaternion sample from IMU and returns array.
		Requires FISC PCB connected through USB directly to computer.
		Change imuCMD in top to correct serial port.
	'''
	p = subprocess.Popen(imuCMD, stdout=subprocess.PIPE, shell=True)
	out, err = p.communicate()
	result = out.decode()

	if len(result)>20:
		sep = re.findall(r"[+-]? *(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", result)
		w = float(sep[1])
		x = float(sep[2])
		y = float(sep[3])
		z = float(sep[4])
		quatArr = np.array([w, x, y, z])
		return w, x, y, z

def updatePlot(roll, pitch, heading):
	''' Updates VPython visualizer with capsule and vectors.
		Inputs:
			[roll, pitch, heading] in degrees
	'''
	roll = np.deg2rad(roll)
	pitch = np.deg2rad(pitch)
	heading = np.deg2rad(heading)
	rate(50) ## Plot update rate
	k = vector(cos(heading)*cos(pitch), sin(pitch),sin(heading)*cos(pitch)) ## Forward pointing vector for capsule ("roll")


	y=vector(0,1,0) ## Vertical vector
	s=cross(k,y)
	v=cross(s,k)
	vrot=v*cos(roll)+cross(k,v)*sin(roll)


	## Find the vertical vector for capsule
	## Vector for roll (forward axis)
	rVec = np.array([np.cos(heading)*np.cos(pitch), np.sin(pitch), np.sin(heading)*np.cos(pitch)]) ## Forward pointing vector for capsule ("roll")
	z = np.array([0,1,0]) # Fixed vertical vector
	sideVec = np.cross(rVec, z) # Capsule side vector, perpendicular to roll vector and fixed vertical vector ("pitch")
	v = np.cross(sideVec, rVec) # Capsule vertical vector, perpendicular to side (pitch) vector and roll vector

	vRotation = v*np.cos(roll)+np.cross(rVec,v)*np.sin(roll) # Rotate v to get fixed to capsule orientation ("yaw")

	## Find horizontal projection of capsules vertical vector (which points towards water current due to inclination)
	# Then find angle between this vector and "north" in fixed frame of reference
	vRotation_hproj = np.array([vRotation[0], 0, vRotation[2]]) # Horizontal projection of vertical
	vRotation_hproj = vRotation_hproj / np.linalg.norm(vRotation_hproj) # To get unit vector
	currentVec = np.dot(vRotation_hproj, np.array([1,0,0])) # Dot product of current direction vector and north
	currentAngle = np.rad2deg(np.arccos(currentVec)) # Angle between current direction vector and north

	# Must account for 360 degrees of rotation. Since the current angle only describes
	# difference between two vectors, max is 180. Therefore, use horizontal projection
	# to determine when current direction is larger than 180
	if vRotation_hproj[2] < 0:
		currentAngle = 360 - currentAngle

	res_key, res_val = min(headingDict.items(), key=lambda x: abs(currentAngle - x[1]))
	#print(res_key, res_val, sep=', ')
	#print("WATER CURRENT DIR:", currentAngle)
	#print("WATER CURRENT rounded:", res_val, "Current heading:", res_key)

	currentVec = vector(vrot.x, 0, vrot.z) ## from vertical vector
	currentArrow.axis = currentVec
	currentArrow.length=3

	headingVec = vector(cos(heading)*cos(pitch), 0, sin(heading)*cos(pitch))

	currentVectemp = np.array([currentVec.x, currentVec.y, currentVec.z])
	unit_currentVec = currentVectemp / np.linalg.norm(currentVectemp)

	dot_product = np.dot(unit_currentVec, np.array([1,0,0])) ## to get angle between north and current direction

	angle = np.rad2deg(np.arccos(dot_product))
	if currentVec.z < 0:
		angle = 360 - angle
	print("\n")

	### INCLINATION works but is slower ###
	#vrot_arr = np.array([vrot.x, vrot.y, vrot.z]) / np.linalg.norm(np.array([vrot.x, vrot.y, vrot.z]))
	#vert_dot = np.dot(vrot_arr, np.array([0,1,0]))
	#inclinationtest = np.rad2deg(np.arccos(vert_dot))
	#print("\n New inclination:", inclinationtest)

	frontArrow.axis=k
	sideArrow.axis=cross(k,vrot)
	upArrow.axis=vrot
	myObj.axis=k
	myObj.up=vrot
	sideArrow.length=-4
	frontArrow.length=4
	upArrow.length=4

def plotIMUData(files, ace):
	''' Visualizes directional vertical inclination over time based on input files.
	'''
	from tools.acousticProcessing import butterworth_LP_filter
	timeStamps = []
	inclinationArr = []
	currentAngleArr = []
	rollArr = []
	pitchArr = []
	headingArr = []

	for file in files:
		hhmmss = str(re.findall('[0-9]{2}:[0-9]{2}:[0-9]{2}', file))[2:-2]
		timeStamp = str(re.findall('[0-9]{2}:[0-9]{2}:[0-9]{2}.?[0-9]{5}', file))[2:-2]

		data=np.load(file, allow_pickle=True)
		imuData = data['IMU']
		print("File:", file)


		if imuData.shape[0] == 4:
			for i, imuSample in enumerate(imuData):

				roll = float(imuSample[0])
				pitch = float(imuSample[1])
				heading = float(imuSample[2])
				#roll-= 0.43# 0.556 ## Calib test roll
				#pitch -= 0.8#2.7933 ## Calib test pitch

				inclination, currentAngle = inclination_current(roll, pitch, heading)
				if inclination > 10:
					## Measurements during manual handling with extrene inclination are ignored ##
					continue
				else:
					inclinationArr.append(inclination)
					currentAngleArr.append(currentAngle)
					temp_timestamp = datetime.datetime.strptime(timeStamp, '%H:%M:%S.%f')
					timeStamps.append(str(temp_timestamp)[-15:-7])
					rollArr.append(roll)
					pitchArr.append(pitch)
					headingArr.append(heading)
		else:
			roll = imuData[0]
			pitch = imuData[1]
			heading = imuData[2]
			#roll-=0.556 ## Calib test roll
			#pitch += 2.7933 ## Calib test pitch

			inclination, currentAngle = inclination_current(roll, pitch, heading)

			if inclination > 10:
				## Measurements during manual handling with extrene inclination are ignored ##
				continue
			else:
				inclinationArr.append(inclination)
				currentAngleArr.append(currentAngle)
				temp_timestamp = datetime.datetime.strptime(timeStamp, '%H:%M:%S.%f')
				timeStamps.append(str(temp_timestamp)[-15:-4])
				rollArr.append(roll)
				pitchArr.append(pitch)
				headingArr.append(heading)
				#print("Timestamp:", str(temp_timestamp)[-15:-4])

		#inclinationArr.append(inclination)
		#print("\n\r Inclination:", inclination)
		#currentAngleArr.append(currentAngle)
	'''
	### TESTING ###

	#rollArr-=np.mean(rollArr)
	#pitchArr -= np.mean(pitchArr)

	for i, val in enumerate(rollArr):
		inclination, currentAngle = inclination_current(rollArr[i], pitchArr[i], headingArr[i])
		if inclination > 10:
			## Measurements during manual handling are ignored ##
			continue
		else:

			inclinationArr.append(inclination)
			currentAngleArr.append(currentAngle)


	### TESTING END ###
	'''
	current_filtered = butterworth_LP_filter(currentAngleArr, 4, 100, 2)

	currentFig, currentAx = plt.subplots(1, figsize=(10,7))
	currentAx.plot(timeStamps, currentAngleArr, label='Raw Estimates')
	currentAx.plot(timeStamps, current_filtered, label='Filtered Estimates', linewidth=3)

	currentAx.set_xlabel("Time [HH:MM:SS]", fontsize=14)
	currentAx.set_ylabel("Water Current Source Direction [deg]", fontsize=14)
	currentAx.xaxis.set_major_locator(plt.MaxNLocator(10))


	inclination_filtered = butterworth_LP_filter(inclinationArr, 4, 100, 2)

	inclinationFig, inclinationAx = plt.subplots(1, figsize=(10,7))
	inclinationAx.plot(timeStamps, inclinationArr, label='Raw Estimates')
	inclinationAx.plot(timeStamps, inclination_filtered, label='Filtered Estimates')
	#inclinationAx.plot(timeStamps, rollArr, label='roll')
	#inclinationAx.plot(timeStamps, pitchArr, label='pitch')
	plt.title("Capsule Vertical Inlication", fontsize=16)
	inclinationAx.set_xlabel("Time [HH:MM:SS]", fontsize=14)
	inclinationAx.set_ylabel("Inclination Angle [deg]", fontsize=14)
	inclinationAx.xaxis.set_major_locator(plt.MaxNLocator(10))
	plt.tight_layout()

	plt.legend(loc='upper right')

	if ace:
		plt.axvline(x=26.2,color='r', label='axvline - % of full height', lw=4.7, alpha=0.5)
		plt.title("Estimated Water Current Direction at Rataren II", fontsize=16)
		#.savefig(os.getcwd()+'/plots/WaterCurrentDirection_ACE.pdf')
	else:
		plt.title("Estimated Water Current Direction at Sinkaberg Hansen (Rørvik)", fontsize=16)
		#plt.savefig(os.getcwd()+'/plots/WaterCurrentDirection_SH.pdf')
	plt.tight_layout()
	plt.show()
	quit()

def inclination_current(roll, pitch, heading):
	''' Function for estimating directional vertical inclination.
		Inputs:
			- [roll, pitch, heading] (in degrees)
		Outputs:
			- [inclination, heading direction] (in degrees)

	'''
	roll = np.deg2rad(roll)
	pitch = np.deg2rad(pitch)
	heading = np.deg2rad(heading)

	## Find the vertical vector for capsule ##
	rVec = np.array([cos(heading)*cos(pitch), sin(pitch),sin(heading)*cos(pitch)]) #Roll (forward) vector
	z = np.array([0,1,0]) #Vertical vector, fixed
	sideVec = np.cross(rVec, z) # Side vector, perpendicular to roll vector and vertical vector
	v = np.cross(sideVec, rVec) # Vertical vector, perpendicular to side (pitch) vector and roll vector
	vRotation = v*np.cos(roll)+np.cross(rVec,v)*np.sin(roll) # Vertical vector fixed to capsule orientation

	## Find horizontal projection of capsules vertical vector (which points towards water current due to inclination) ##
	## Then find angle between this vector and "north" from fixed frame of reference ##
	vRotation_hproj = np.array([vRotation[0], 0, vRotation[2]]) # Horizontal projection of vertical vector
	vRotation_hproj = vRotation_hproj / np.linalg.norm(vRotation_hproj) # To get unit vector
	currentVec = np.dot(vRotation_hproj, np.array([1,0,0])) # Dot product of current dir vector and north
	currentAngle = np.rad2deg(np.arccos(currentVec)) # Angle between current vector and north

	# Must account for 360 degrees rotation. Since the current angle only describes
	# difference between two vectors, max is 180. Therefore, use horizontal projection
	# to determine when current direction is larger than 180
	if vRotation_hproj[2] < 0:
		currentAngle = 360 - currentAngle

	## Find inclination/tilt angle from trigonometry ##
	rollsq = np.tan(roll)*np.tan(roll)
	pitchsq = np.tan(pitch)*np.tan(pitch)
	inclination = np.degrees(np.arctan(np.sqrt(rollsq+pitchsq)))

	return round(inclination, 2), round(currentAngle, 2)

if __name__=='__main__':
	init()
	while(True):
		w, x, y, z = getQuat()
		roll, pitch, heading = quaternionToYawPitchRoll(w, x, y, z)
		print("Roll:",roll,"Pitch:",pitch,"Heading:",heading)

		updatePlot(roll,pitch,heading)

		inclination, currentDirection = inclination_current(roll, pitch, heading)
		r_currentHeading, r_currentDirection = min(headingDict.items(), key=lambda x: abs(currentDirection - x[1]))


		print("Inclination: ", inclination, "Current direction:", currentDirection, "Current heading:", r_currentHeading)
		print("\n")
