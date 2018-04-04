import numpy as np
from scipy import optimize as opt
import math



#该函数用于将旋转矩阵转换成向量形式时要使用的
def max_norm(R_plus):
	val_1=np.linalg.norm(R_plus[:,0])
	val_2=np.linalg.norm(R_plus[:,1])	
	val_3=np.linalg.norm(R_plus[:,1])
	x=np.array([val_1,val_2,val_3])
	re=np.where(x==np.max(x))
	print(re)
	v=R_plus[:,re].T

	return v

#将旋转矩阵转换成向量形式
def to_rodrigues_vec(R):
	R=np.array(R)
	p=np.array([
		[R[2,1]-R[2,1]],
		[R[0,2]-R[2,0]],
		[R[1,0]-R[0,1]]
		])
	c=0.5*(np.trace(R)-1)

	if np.linalg.norm(p)==0:
		if c==1:
			rou=np.array([0,0,0])
		elif c==-1:
			R_plus=R+np.eye([3,3],float)
			v=max_norm(R_plus)
			u=v/np.linalg.norm(v)
			if (u[0]<0) | ((u[0]==0)&(u[1]<0))|((u[0]==0)&(u[1]==0)&(u[2]<0)):
				u=-u
			rou=np.pi*u
		else:
			rou=[]
	else:
		u=p/np.linalg.norm(p)
		zeta=math.atan2(np.linalg.norm(p),c)
		rou=zeta*u
	return rou.transpose()

#将旋转矩阵转换成原形式
def to_rotation_mat(rou):
	zeta=np.linalg.norm(rou)
	rou=rou/zeta
	W=np.array([
		[0,-rou[2],rou[1]],
		[rou[2],0,-rou[0]],
		[-rou[1],rou[0],0]
		])
	R=np.eye([3,3],float)+W*np.sin(zeta)+np.dot(W,W)*(1-np.cos(zeta))
	return R



def compose_para_vec(intrinsics,distortion,extrinsics):
	a=np.array([intrinsics[0,0],intrinsics[1,1],intrinsics[0,1],intrinsics[0,2],intrinsics[1,2],distortion[0],distortion[1]])
	P=[]
	P.append(a)
	for i in range(len(extrinsics)):

		R=extrinsics[i][:,0:3]

		t=extrinsics[i][:,3]
		rou=to_rodrigues_vec(R)
		#print(rou)
		#print(t)

		P.append(rou)
		P.append(t)

	return np.array(P).reshape((-1,1))

def decompose_para_vec(P):
	A=np.array([
		[P[0],P[2],P[3]],
		[0,P[1],P[4]],
		[0,0,1]
		])
	k=np.array([P[5],P[6]])
	W_final=np.array([])
	for i in range(16):
		m=7+6*i
		rou=np.array([P[m],P[m+1],P[m+2]])
		t=np.array([P[m+3],P[m+4],P[m+5]]).T
		R=to_rotation_mat(rou)
		W=np.array([R,t])
		W_final=np.append(W_final,W)
	return A,k,W_final

def val(P,data):
	[real_corners,img_corners]=data
	corners_size=int(real_corners.size/2)
	Y=[]

	A=np.array([
		[P[0],P[2],P[3]],
		[0,P[1],P[4]],
		[0,0,1]
		])
	for i in range(16):
		m=7+i*6
		rou=np.array([P[m],P[m+1],P[m+2]])
		t=np.array([P[m+3],P[m+4],P[m+5]]).T
		R=to_rotation_mat(rou)
		W=np.array([R,t])
		for j in range(corners_size):
			x=real_corners[j][0]
			y=real_corners[j][1]
			homo_XY = np.transpose(np.array([x, y,0, 1]))

			[u, v,w] = np.dot(A,np.dot(W, homo_XY))
			u=u/w
			v=v/w
        
			Y.append(u)
			Y.append(v)
	return np.array(Y)

def jac(P,data):
	[real_corners,img_corners]=data
	print('P_length')
	print(len(P))
	Y=val(P,data).T
	J=[]
	for k in range(len(P)):
		J_temp=np.gradient(Y,P[k])
		J.append(J_temp)
	print('jacob')
	print(J)	
	return np.array(J).T

def refine_all(P,real_corners,img_corners):
	X=np.array([])
	Y=np.array([])
	print(len(img_corners))
	for i in range(len(img_corners)):
		X=np.append(X,real_corners.reshape((-1,1)))
		Y=np.append(Y,img_corners[i].reshape((-1,1)))
	refined_P=opt.root(
		val,
		P,
		jac=jac,
		args=[X,Y],
		method='lm'
		).x
	return decompose_para_vec(P)


