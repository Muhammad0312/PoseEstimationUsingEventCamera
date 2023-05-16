#!/usr/bin/python3

from homo_decomposition import *
from homo_generation2 import *
# from create_mosaic import *
from homography_eklt import *

# # underwater poster
# K = np.array([[285.78, 0, 181.13],
#                   [0, 286.11, 120.86],
#                   [0, 0, 1]])

# TV poster
K = np.array([[618.9265, 0, 179.2565],
              [0, 618.2467, 115.8944],
              [0,    0,         1   ]])


# # # Compute homographies first
dataset = 'tv_poster'
path = dataset+'/images.txt'

# # # num_images = 1035 # Underwater poster
# # # num_images = 1440 # TV poster
# num_images = 1035 # Underwater poster
# num_images = 1440 # TV poster
num_images = -1
homography_list = get_homography(dataset,num_images,draw=False,scale = False)


# # print(homography_list)

homography_list = np.array(homography_list)

# homography list = [1H2 2H3 3H4 ...]

wHp, pH1 =  get_scale('tv_poster/poster_gray.jpg', 'tv_poster/images/0.png', 350,710,False)
# wHp, pH1 =  get_scale('tv_poster/poster_gray.jpg', 'tv_poster/poster_half_horz.jpg', 3.50,7.10,False)

wH1 = wHp @ pH1
print(wHp, pH1)
print(wH1)

homography_list = np.insert(homography_list, 0, wH1, axis=0)
# homography list = [wHp pH1 1H2 2H3 3H4 ...]

plot_camera_trajectory(homography_list, K)

folder_name = 'event_files'
filename = 'slider_close_timescaled.txt'
path = folder_name + '/' +filename

frame_dict = FramesFromFeatures(path)
all_homographies = HomographyEkltFeatures(frame_dict)
plot_camera_trajectory(all_homographies, K)
# get scale (first image, poster image)
# wHp, pH1 =  get_scale('tv_poster/poster_gray.jpg', 'tv_poster/images/0.png', 350,710)

# print(wHp,pH1)