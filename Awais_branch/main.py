#!/usr/bin/python3

from homo_decomposition import *
from homo_generation import *
from homography_eklt import *
# from create_mosaic import *

# Compute homographies first
dataset = 'slider_close'
num_images = -1
homography_list = get_homography(dataset,num_images,draw=False,scale = False)


# Compute and display trajectories
calib_params = get_camera_parameters(dataset)

K = np.array([[calib_params[0], 0, calib_params[2]],
                        [0, calib_params[1], calib_params[3]],
                        [0, 0, 1]])

# # Decompose homography matrix
# # retval, R1, t1, n = cv2.decomposeHomographyMat(homography_list, K)


plot_camera_trajectory(homography_list,K)


folder_name = 'event_files'
filename = 'slider_close_timescaled.txt'
path = folder_name + '/' +filename

frame_dict = FramesFromFeatures(path)
all_homographies = HomographyEkltFeatures(frame_dict)
# print(np.array(all_homographies))

# Decompose homography matrix
# retval, R1, t1, n = cv2.decomposeHomographyMat(all_homographies, K)


plot_camera_trajectory(np.array(all_homographies),K)