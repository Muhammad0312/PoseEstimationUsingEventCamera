#!/usr/bin/python3

from homo_decomposition import *
from homo_generation import *
# from create_mosaic import *

K = np.array([[285.78, 0, 181.13],
                  [0, 286.11, 120.86],
                  [0, 0, 1]])

# # Compute homographies first
dataset = 'poster_dataset'
path = dataset+'/images.txt'
num_images = 1035
# num_images = 100
homography_list = get_homography(dataset,path,num_images,draw=False,scale = False)

# print(homography_list)

# # Compute and display trajectories
plot_camera_trajectory(homography_list,K)

# Create a mosaic
# create_mosaic(homography_list,num_images,path='poster_dataset/images/')
