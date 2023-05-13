#!/usr/bin/python3

from homo_decomposition import *
from homo_generation2 import *
# from create_mosaic import *

# # underwater poster
# K = np.array([[285.78, 0, 181.13],
#                   [0, 286.11, 120.86],
#                   [0, 0, 1]])

# TV poster
K = np.array([[618.9265, 0, 179.2565],
                  [0, 618.2467, 115.8944],
                  [0, 0, 1]])


# # Compute homographies first
dataset = 'slider_close'
path = dataset+'/images.txt'
# num_images = 1035 # Underwater poster
# num_images = 1440 # TV poster
num_images = 200
homography_list = get_homography(dataset,num_images,draw=False,scale = False)

# print(homography_list)

# get scale (first image, poster image)
# s =  get_scale('tv_poster/images/0.png', 'poster_dataset/poster_gray.jpg')

# print(homography_list[0])
# homography_list[0] = s @ homography_list[0]
# print(homography_list[0])
# print(s)

# # Compute and display trajectories
# plot_camera_trajectory(homography_list,K)

# Create a mosaic
# create_mosaic(homography_list,num_images,path='poster_dataset/images/')
