import numpy as np
from scipy import optimize as opt

#求解超定方程组之前的标准化数据，通过z-score标准化
def get_normalisation_matrix(corners):
	avg_x = corners[:,0].mean()
	avg_y = corners[:,1].mean()

	std_x = np.sqrt(2 / pow(corners[0].std(),2))
	std_y = np.sqrt(2 / pow(corners[1].std(),2))

	return np.matrix([
		[std_x,   0,   -std_x * avg_x],
		[0,   std_y,   -std_y * avg_y],
		[0,     0,              1]
	])

#通过SVD来求解超定方程组，得到homography
def estimate_homography(real_corners,img_corners):
	real_corners_normalisation_matrix=get_normalisation_matrix(real_corners)
	img_corners_normalisation_matrix=get_normalisation_matrix(img_corners)

	M = [] #normalized matrix consists of real_corners and img_corners

	corners_size=int(real_corners.size/2)

	for corner_num in range(corners_size):
		homogeneous_real_corners=np.array([
			real_corners[corner_num,0],#代码有问题的话注意下这里
			real_corners[corner_num,1],
			1
		])

		homogeneous_img_corners=np.array([
			img_corners[corner_num,0],#代码有问题的话注意下这里
			img_corners[corner_num,1],
			1
		])

		#得到世界坐标和图像坐标 标准化后的齐次坐标
		normalized_homo_real_corners=np.dot(real_corners_normalisation_matrix,homogeneous_real_corners)
		normalized_homo_img_corners=np.dot(img_corners_normalisation_matrix,homogeneous_img_corners)


		#构建MH=0的形式，通过最小二乘解-SVD方法来求解H
		M.append(np.array([
			normalized_homo_real_corners.item(0),normalized_homo_real_corners.item(1),1,
			0,0,0,
			-normalized_homo_real_corners.item(0)*normalized_homo_img_corners.item(0),-normalized_homo_real_corners.item(1)*normalized_homo_img_corners.item(0),-normalized_homo_img_corners.item(0)
		]))

		M.append(np.array([
			0,0,0,
			normalized_homo_real_corners.item(0),normalized_homo_real_corners.item(1),1,
			-normalized_homo_real_corners.item(0)*normalized_homo_img_corners.item(1),-normalized_homo_real_corners.item(1)*normalized_homo_img_corners.item(1),-normalized_homo_img_corners.item(1)
		]))

	#SVD分解，V的最后一项是方程MH=0的解
	U,S,Vh=np.linalg.svd(np.array(M).reshape((corners_size*2,9)))
	L=Vh[-1]
	H=L.reshape(3,3)

	#denormalization H,此处也有一些问题：推导的公式和案例中的不一样，我先按自己推导的来写，注意修改
	denormalised_H=np.dot(
			np.dot(
				np.linalg.inv(img_corners_normalisation_matrix),
				H
				),
				real_corners_normalisation_matrix
		)
	denormalised_H=denormalised_H/denormalised_H[-1,-1]

	modified_H=np.array([[denormalised_H[0,1],denormalised_H[0,0],denormalised_H[0,2]],
		[denormalised_H[1,1],denormalised_H[1,0],denormalised_H[1,2]],
		[denormalised_H[2,1],denormalised_H[2,0],denormalised_H[2,2]]
		])

	return modified_H

#定义jacob矩阵用于LM对于H的优化
def jacob(homography,data):
	[real_corners,img_corners]=data
	J=[]
	corners_size=int(real_corners.size/2)

	for i in range(corners_size):
		x=real_corners[i][0]
		y=real_corners[i][1]

		s_x = homography.item(0) * x + homography.item(1) * y + homography.item(2)
		s_y = homography.item(3) * x + homography.item(4) * y + homography.item(5)
		w = homography.item(6) * x + homography.item(7) * y + homography.item(8)
        
		J.append(
			np.array([
				x / w, y / w, 1/w,
				0, 0, 0,
				(-s_x * x) / (w*w), (-s_x * y) / (w*w), -s_x / (w*w)
			])
		)

		J.append(
			np.array([
				0, 0, 0,
				x / w, y / w, 1 / w,
				(-s_y * x) / (w*w), (-s_y * y) / (w*w), -s_y / (w*w)
			])
		)

	return np.array(J)

#定义LM算法的cost function
def cost(homography,data):
	[real_corners,img_corners]=data
	Y=[]
	corners_size=int(real_corners.size/2)
	for i in range(corners_size):
		x=real_corners[i][0]
		y=real_corners[i][1]
		w = homography.item(6) * x + homography.item(7) * y + homography.item(8)
        
		M = np.array([
			[homography.item(0), homography.item(1), homography.item(2)],
			[homography.item(3), homography.item(4), homography.item(5)]
		])
		homo_XY = np.transpose(np.array([x, y, 1]))
		[u, v] = (1/w) * np.dot(M, homo_XY)
        
		Y.append(u)
		Y.append(v)
	return np.array(Y)


#LM算法优化H的值
def refine_homography(homography, real_corners,img_corners):
	refined_homography=opt.root(
		cost,
		homography,
		jac=jacob,
		args=[real_corners,img_corners],
		method='lm'
		).x
	refined_homography=refined_homography/refined_homography[-1]
	return refined_homography



def compute_homography(data):
	real_corners=data['real']
	refined_homographies = []

	for i in range(len(data['sensed'])):
		img_corners=data['sensed'][i]

		estimated_homography = estimate_homography(real_corners,img_corners)

		refined_homography= refine_homography(estimated_homography,real_corners,img_corners)
		refined_homographies.append(estimated_homography)

	return np.array(refined_homographies)