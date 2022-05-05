import numpy as np
from vpython import *
import time
import re
import datetime
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os


from tools.acousticProcessing import butterworth_LP_filter

headingDict = {"N":0, "NNE":22.5, "NE":45, "ENE":67.5, "E":90,\
			   "ESE":112.5, "SE":135, "SSE":157.5, "S":180, \
			   "SSW":202.5, "SW":225, "WSW":247.5, "W":270, \
			   "WNW":292.5, "NW":315, "NNW":337.5}


def init():
	global kArrow, currentArrow, frontArrow, upArrow, sideArrow, myObj
	scene.range=5
	scene.background=color.yellow
	toRad=2*np.pi/360
	toDeg=1/toRad
	scene.forward=vector(-1,-1,-1)

	scene.width=1200
	scene.height=1080

	xarrow=arrow(lenght=2, shaftwidth=.1, color=color.red,axis=vector(1,0,0))
	yarrow=arrow(lenght=2, shaftwidth=.1, color=color.green,axis=vector(0,1,0))
	zarrow=arrow(lenght=4, shaftwidth=.1, color=color.blue,axis=vector(0,0,-1))

	frontArrow=arrow(length=4,shaftwidth=.1,color=color.purple,axis=vector(1,0,0))
	upArrow=arrow(length=1,shaftwidth=.1,color=color.magenta,axis=vector(0,1,0))
	sideArrow=arrow(length=2,shaftwidth=.1,color=color.orange,axis=vector(0,0,1))

	kArrow=arrow(length=5,shaftwidth=.1,color=color.orange,axis=vector(0,0,1))


	#incVec = arrow(length=3, shaftwidth=0.05, color=color.black, axis=vector(1,0,-1))
	headingArrow = arrow(length=3, shaftwidth=0.05, color=color.black, axis=vector(1,0,-1))
	currentArrow = arrow(length=3, shaftwidth=0.05, color=color.yellow, axis=vector(1,0,-1))


	IMU=box(lenght=0.1,width=0.3,height=.1,pos=vector(0,0.2,-0.5),color=color.black)
	PCB = cylinder(pos=vector(0,0,0), axis=vector(0,0.2,0), radius=1, opacity=.8, color=color.green)
	myObj=compound([PCB,IMU])

def plotIMUData(files, ace):
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


		if imuData.shape[0] == 4:
			for i, imuSample in enumerate(imuData):

				roll = float(imuSample[0])
				pitch = float(imuSample[1])
				heading = float(imuSample[2])

				#roll-= 0.43# 0.556 ## Calib test roll
				#pitch -= 0.8#2.7933 ## Calib test pitch

				inclination, currentAngle = inclination_current(roll, pitch, heading)
				if inclination > 10:
					## Measurements during manual handling are ignored ##
					continue
				else:
					inclinationArr.append(inclination)
					currentAngleArr.append(currentAngle)
					temp_timestamp = datetime.datetime.strptime(timeStamp, '%H:%M:%S.%f')
					timeStamps.append(str(temp_timestamp)[-15:-7])
					rollArr.append(roll)
					pitchArr.append(pitch)
					headingArr.append(heading)
					#print("Timestamp:", str(temp_timestamp)[-15:-4])
		else:
			roll = imuData[0]
			pitch = imuData[1]
			heading = imuData[2]
			#roll-=0.556 ## Calib test roll
			#pitch += 2.7933 ## Calib test pitch

			inclination, currentAngle = inclination_current(roll, pitch, heading)

			if inclination > 10:
				## Measurements during manual handling are ignored ##
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
	### TESTING
	'''
	rollArr-=np.mean(rollArr)
	pitchArr -= np.mean(pitchArr)

	for i, val in enumerate(rollArr):
		inclination, currentAngle = inclination_current(rollArr[i], pitchArr[i], headingArr[i])
		if inclination > 10:
			## Measurements during manual handling are ignored ##
			continue
		else:

			inclinationArr.append(inclination)
			currentAngleArr.append(currentAngle)


	### TESTING END
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
	inclinationAx.plot(timeStamps, rollArr, label='roll')
	inclinationAx.plot(timeStamps, pitchArr, label='pitch')
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
	#print("Roll:", np.degrees(roll))
	#print("Pitch:", np.degrees(pitch))
	#print("Heading:", np.degrees(heading))
	#print("Inclination:", inclination)
	#input("Wait")

	#r_currentHeading, r_currentDirection = min(headingDict.items(), key=lambda x: abs(currentDirection - x[1]))

	return round(inclination, 2), round(currentAngle, 2)

def updatePlot(roll, pitch, heading):
	roll = np.deg2rad(roll)
	pitch = np.deg2rad(pitch)
	heading = np.deg2rad(heading)
	rate(50)
	k = vector(cos(heading)*cos(pitch), sin(pitch),sin(heading)*cos(pitch))
	kArrow.axis = vector(cos(heading)*cos(pitch), sin(pitch),sin(heading)*cos(pitch))
	kArrow.length=5
	#print("k",k)
	#quit()
	y=vector(0,1,0)
	s=cross(k,y)
	v=cross(s,k)
	vrot=v*cos(roll)+cross(k,v)*sin(roll)


	### TESTING ###

	## Find the vertical vector for capsule
	## Vector for roll (forward axis)
	rVec = np.array([np.cos(heading)*np.cos(pitch), np.sin(pitch), np.sin(heading)*np.cos(pitch)]) #Roll (forward) vector
	z = np.array([0,1,0]) #Vertical vector, fixed frame of referecnce
	sideVec = np.cross(rVec, z) # Side vector, perpendicular to roll vector and vertical vector
	v = np.cross(sideVec, rVec) # Vertical vector, perpendicular to side (pitch) vector and roll vector
	vRotation = v*np.cos(roll)+np.cross(rVec,v)*np.sin(roll) # Vertical vector fixed to capsule orientation

	## Find horizontal projection of capsules vertical vector (which points towards water current due to inclination)
	# Then find angle between this vector and "north" from fixed frame of reference
	vRotation_hproj = np.array([vRotation[0], 0, vRotation[2]]) # Horizontal projection of vertical
	vRotation_hproj = vRotation_hproj / np.linalg.norm(vRotation_hproj) # To get unit vector
	currentVec = np.dot(vRotation_hproj, np.array([1,0,0])) # Dot product of current dir vector and north
	currentAngle = np.rad2deg(np.arccos(currentVec)) # Angle between current vector and north

	# Must account for 360 deg rotation. Since the current angle only describes
	# difference between two vectors, max is 180. Therefore, use horizontal projection
	# to determine when current direction is larger than 180
	if vRotation_hproj[2] < 0:
		currentAngle = 360 - currentAngle

	res_key, res_val = min(headingDict.items(), key=lambda x: abs(currentAngle - x[1]))
	print(res_key, res_val, sep=', ')
	print("WATER CURRENT DIR:", currentAngle)
	print("WATER CURRENT rounded:", res_val, "Current heading:", res_key)

	### TESTING END ###

	currentVec = vector(vrot.x, 0, vrot.z) ## from vertical vector
	currentArrow.axis = currentVec
	currentArrow.length=1

	headingVec = vector(cos(heading)*cos(pitch), 0, sin(heading)*cos(pitch))
	#headingArrow.axis = headingVec

	#headingArrow.length=1¨

	#currentDirection = np.rad2deg(headingVec.diff_angle(currentVec))
	#print("DIRECTION:", currentDirection)
	#print("\n current x:",currentVec.x,"current z:",currentVec.z)

	currentVectemp = np.array([currentVec.x, currentVec.y, currentVec.z])
	unit_currentVec = currentVectemp / np.linalg.norm(currentVectemp)
	#unit_headingVec = headingVectemp / np.linalg.norm(headingVectemp)

	#dot_product = np.dot(unit_currentVec, unit_headingVec)
	dot_product = np.dot(unit_currentVec, np.array([1,0,0])) ## to get angle between north and current direction

	angle = np.rad2deg(np.arccos(dot_product))
	if currentVec.z < 0:
		angle = 360 - angle
	#print("\n CURRENT DIRECTION:", angle)
	#print("CURRENT VEC Z:", np.rad2deg(currentVec.z))
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
	sideArrow.length=-2
	frontArrow.length=4
	upArrow.length=1

if __name__=='__main__':
	init()
	while(True):
		roll = np.random.randint(90)
		pitch = np.random.randint(90)
		heading = np.random.randint(360)
		#updatePlot(2.286, 0.55, 36)
		time.sleep(0.5)
