import parser
import Compute_Homography
import numpy as np
import intrinsics
import extrinsics
import distortion
import Read_Corners
import overall_refinement

def parse_data(basepath="C:\\Users\\zluak\\Desktop\\lzj_implementation\\data\\corners_", ext=".dat"):
	sensed = []
	for i in range(1, 6):
		sensed.append(np.loadtxt(basepath + str(i) + ext).reshape((256, 2)))

	return {
		'real': np.loadtxt(basepath + "real" + ext).reshape((256, 2)),
		'sensed': sensed
	}

#data=parse_data()
data=Read_Corners.read_data()


homographies=Compute_Homography.compute_homography(data)
print("homographies")
print(homographies)

intrinsic_paras=intrinsics.get_camera_intrinsics(homographies)
print("intrinsic_paras")
print(intrinsic_paras)

extrinsic_paras=extrinsics.get_camera_extrinsics(intrinsic_paras, homographies)
print("extrinsic_paras")
print(extrinsic_paras)

distortion_paras = distortion.estimate_lens_distortion(
		intrinsic_paras,
		extrinsic_paras,
		data["real"],
		data["sensed"]
    )

print("distortion_paras")
print(distortion_paras)

P=overall_refinement.compose_para_vec(intrinsic_paras,distortion_paras,extrinsic_paras)
print('P')
print(P)
print("p1")
print(P[0])
print(P[1])
A,k,W=overall_refinement.refine_all(P,data["real"],data["sensed"])
print('refined_intrinsics')
print(A)

