import numpy as np
from vpython import *
import time
import re
import datetime
import matplotlib.pyplot as plt
#from toolsimport tools.acousticProcessing

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

def plotData(files):
    timeStamps = []
    inclinationArr = []
    currentAngleArr = []
    for file in files:
        hhmmss = str(re.findall('[0-9]{2}:[0-9]{2}:[0-9]{2}', file))[2:-2]
        timeStamp = str(re.findall('[0-9]{2}:[0-9]{2}:[0-9]{2}.?[0-9]{5}', file))[2:-2]
        #timeStamps.append(datetime.datetime.strptime(timeStamp, '%H:%M:%S.%f'))

        #print("file:", file)
        data=np.load(file, allow_pickle=True)
        imuData = data['IMU']

        if imuData.shape[0] == 4:
            for i, imuSample in enumerate(imuData):
                #print(sample)
                roll = imuSample[0]
                pitch = imuSample[1]
                heading = imuSample[2]
                timeStamps.append(datetime.datetime.strptime(timeStamp, '%H:%M:%S.%f'))
                inclination, currentAngle = inclination_current(roll, pitch, heading)
                if inclination > 20:
                    inclinationArr.append(inclinationArr[i-1])
                    currentAngleArr.append(currentAngleArr[i-1])
                else:
                    inclinationArr.append(inclination)
                    currentAngleArr.append(currentAngle)
                #print(roll, pitch, heading)
                #print("Inclination:", inclination)

        else:
            roll = imuData[0]
            pitch = imuData[1]
            heading = imuData[2]
            timeStamps.append(datetime.datetime.strptime(timeStamp, '%H:%M:%S.%f'))
            inclination, currentAngle = inclination_current(roll, pitch, heading)
            if inclination > 20:
                inclinationArr.append(inclinationArr[i-1])
                currentAngleArr.append(currentAngleArr[i-1])
            else:
                inclinationArr.append(inclination)
                currentAngleArr.append(currentAngle)
            #inclinationArr.append(inclination)
            #currentAngleArr.append(currentAngle)
            #print(roll, pitch, heading)


        #inclinationArr.append(inclination)
        #print("\n\r Inclination:", inclination)
        #currentAngleArr.append(currentAngle)
    current_filtered = acousticProcessing.butterworth_LP_filter(currentAngleArr, 4, 100, 2)
    plt.plot(timeStamps, currentAngleArr)
    plt.plot(timeStamps, current_filtered)
    plt.title("Estimated water current direction over time")
    plt.show()
    quit()

def inclination_current(roll, pitch, heading):
    roll = np.deg2rad(roll)
    pitch = np.deg2rad(pitch)
    heading = np.deg2rad(heading)

    ## Find the vertical vector for capsule ##
    rVec = np.array([cos(heading)*cos(pitch), sin(pitch),sin(heading)*cos(pitch)]) #Roll (forward) vector
    z = np.array([0,1,0]) #Vertical vector, fixed frame of referecnce
    sideVec = np.cross(rVec, z) # Side vector, perpendicular to roll vector and vertical vector
    v = np.cross(sideVec, rVec) # Vertical vector, perpendicular to side (pitch) vector and roll vector
    vRotation = v*np.cos(roll)+np.cross(rVec,v)*np.sin(roll) # Vertical vector fixed to capsule orientation

    ## Find horizontal projection of capsules vertical vector (which points towards water current due to inclination) ##
    ## Then find angle between this vector and "north" from fixed frame of reference ##
    vRotation_hproj = np.array([vRotation[0], 0, vRotation[2]]) # Horizontal projection of vertical
    vRotation_hproj = vRotation_hproj / np.linalg.norm(vRotation_hproj) # To get unit vector
    currentVec = np.dot(vRotation_hproj, np.array([1,0,0])) # Dot product of current dir vector and north
    currentAngle = np.rad2deg(np.arccos(currentVec)) # Angle between current vector and north

    # Must account for 360 deg rotation. Since the current angle only describes
    # difference between two vectors, max is 180. Therefore, use horizontal projection
    # to determine when current direction is larger than 180
    if vRotation_hproj[2] < 0:
        currentAngle = 360 - currentAngle

    ## Find inclination angle from trigonometry ##
    rollsq = np.tan(roll)*np.tan(roll)
    pitchsq = np.tan(pitch)*np.tan(pitch)
    inclination = np.degrees(np.arctan(np.sqrt(rollsq+pitchsq)))

    #r_currentHeading, r_currentDirection = min(headingDict.items(), key=lambda x: abs(currentDirection - x[1]))

    return inclination, currentAngle

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
