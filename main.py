#!/usr/bin/python3

from homo_decomposition import *
from homo_generation import *
from create_mosaic import *

K = np.array([[285.78, 0, 181.13],
                  [0, 286.11, 120.86],
                  [0, 0, 1]])

# # Compute homographies first
dataset = 'poster_dataset'
path = dataset+'/images.txt'
# num_images = 100
num_images = 1035
homography_list = get_homography(dataset,path,num_images)

print(homography_list)

# # Compute and display trajectories
plot_camera_trajectory(homography_list,K)

# create mosaic using homhgraphies
# create_mosaic('poster_dataset/images/0.png',homography_list)