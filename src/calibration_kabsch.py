##################################################################################################
##       License: Apache 2.0. See LICENSE file in root directory.		                      ####
##################################################################################################
##                  Box Dimensioner with multiple cameras: Helper files 					  ####
##################################################################################################
import pyrealsense2 as rs
import calculate_rmsd_kabsch as rmsd
import numpy as np

'''
Adapted from
    Box dimensions calculation using multiple realsense camera
    https://github.com/IntelRealSense/librealsense/tree/master/wrappers/python/examples/box_dimensioner_multicam
'''

"""
  _   _        _                      _____                     _    _                    
 | | | |  ___ | | _ __    ___  _ __  |  ___|_   _  _ __    ___ | |_ (_)  ___   _ __   ___ 
 | |_| | / _ \| || '_ \  / _ \| '__| | |_  | | | || '_ \  / __|| __|| | / _ \ | '_ \ / __|
 |  _  ||  __/| || |_) ||  __/| |    |  _| | |_| || | | || (__ | |_ | || (_) || | | |\__ \
 |_| |_| \___||_|| .__/  \___||_|    |_|    \__,_||_| |_| \___| \__||_| \___/ |_| |_||___/
				 _|                                                                      
"""


def calculate_transformation_kabsch(src_points, dst_points):
	"""
	Calculates the optimal rigid transformation from src_points to
	dst_points
	(regarding the least squares error)

	Parameters:
	-----------
	src_points: array
		(3,N) matrix
	dst_points: array
		(3,N) matrix
	
	Returns:
	-----------
	rotation_matrix: array
		(3,3) matrix
	
	translation_vector: array
		(3,1) matrix

	rmsd_value: float

	"""
	assert src_points.shape == dst_points.shape
	if src_points.shape[0] != 3:
		raise Exception("The input data matrix had to be transposed in order to compute transformation.")
		
	src_points = src_points.transpose()
	dst_points = dst_points.transpose()
	
	src_points_centered = src_points - rmsd.centroid(src_points)
	dst_points_centered = dst_points - rmsd.centroid(dst_points)

	rotation_matrix = rmsd.kabsch(src_points_centered, dst_points_centered)
	rmsd_value = rmsd.kabsch_rmsd(src_points_centered, dst_points_centered)

	translation_vector = rmsd.centroid(dst_points) - np.matmul(rmsd.centroid(src_points), rotation_matrix)

	return rotation_matrix.transpose(), translation_vector.transpose(), rmsd_value



"""
  __  __         _           ____               _                _   
 |  \/  |  __ _ (_) _ __    / ___| ___   _ __  | |_  ___  _ __  | |_ 
 | |\/| | / _` || || '_ \  | |    / _ \ | '_ \ | __|/ _ \| '_ \ | __|
 | |  | || (_| || || | | | | |___| (_) || | | || |_|  __/| | | || |_ 
 |_|  |_| \__,_||_||_| |_|  \____|\___/ |_| |_| \__|\___||_| |_| \__|																	 

"""

class Transformation:
	def __init__(self, rotation_matrix, translation_vector):
		self.pose_mat = np.zeros((4,4))
		self.pose_mat[:3,:3] = rotation_matrix
		self.pose_mat[:3,3] = translation_vector.flatten()
		self.pose_mat[3,3] = 1
	
	def apply_transformation(self, points):
		""" 
		Applies the transformation to the pointcloud
		
		Parameters:
		-----------
		points : array
			(3, N) matrix where N is the number of points
		
		Returns:
		----------
		points_transformed : array
			(3, N) transformed matrix
		"""
		assert(points.shape[0] == 3)
		n = points.shape[1] 
		points_ = np.vstack((points, np.ones((1,n))))
		points_trans_ = np.matmul(self.pose_mat, points_)
		points_transformed = np.true_divide(points_trans_[:3,:], points_trans_[[-1], :])
		return points_transformed
	
	def inverse(self):
		"""
		Computes the inverse transformation and returns a new Transformation object

		Returns:
		-----------
		inverse: Transformation

		"""
		rotation_matrix = self.pose_mat[:3,:3]
		translation_vector = self.pose_mat[:3,3]
		
		rot = np.transpose(rotation_matrix)
		trans = - np.matmul(np.transpose(rotation_matrix), translation_vector)
		return Transformation(rot, trans)
	
