#!/usr/bin/python3

from homo_decomposition import *
from homo_generation import *

K = np.array([[199.092366542, 0, 132.192071378],
                  [0, 198.82882047, 110.712660011],
                  [0, 0, 1]])

# # Compute homographies first
dataset = 'poster_dataset'
path = dataset+'/images.txt'
num_images = 1035
homography_list = get_homography(dataset,path,num_images)

print(homography_list)

# Compute and display trajectories
plot_camera_trajectory(homography_list,K)