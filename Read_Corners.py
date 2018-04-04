import sys
import numpy as np
'''
f = open(r"C:\\Users\\zluak\\Desktop\\stereo_calibration_test\\2018.3.9_calibration\\modified_right\\right_corners.txt")
lines=f.readlines()

corners=np.zeros((200,2,2),dtype=float)
corners_row=0

num=0
for line in lines:
	list=line.strip('\n').split(' ')
	if list==ord(' '):
		num+=1
	#print(list[2])
	corners[corners_row,0,num]=map(float,list[0])
	corners[corners_row,1,num]=map(float,list[1])
	corners_row+=1
	#print(list)
	#corners.append(line.strip())

print(corners)
'''


def read_data():
	coners_num_per_image=48
	img_num=16



	fname_right='C:\\Users\\zluak\\Desktop\\stereo_calibration_test\\2018.03.15\\left\\left_corners.txt'
	f_right=np.loadtxt(fname_right)
	corners_right=np.reshape(f_right,(img_num,coners_num_per_image,2))
	sensed = []
	for i in range(img_num):
		sensed.append(corners_right[i,:,:])


	objp = np.zeros((6*8,2), np.float32)
	objp = np.mgrid[range(50,-10,-10),range(0,80,10)].T.reshape(-1,2)

	return {
		'real': objp,
		'sensed': sensed
	}
