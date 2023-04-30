#!/usr/bin/python3

import numpy as np
import cv2

def show_warped_images(warped_img1,img2):

    # Show stitched image
    warped_img1[0:img2.shape[0], 0:img2.shape[1]] = img2
    cv2.namedWindow('Combined', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Combined', 800, 600)
    cv2.moveWindow('Combined', 1000, 100)
    cv2.imshow('Combined', warped_img1)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def create_mosaic(homographies,num_images,path):

    
    cumulative_homography = np.eye(3)
    cumulative_homographies = [cumulative_homography]
    for H in homographies:
        cumulative_homography = np.dot(cumulative_homography,H)
        cumulative_homographies.append(cumulative_homography)

    cumulative_homographies = np.array(cumulative_homographies)[1:]
    print(cumulative_homographies.shape)

    img1 = cv2.imread(path + '0.png')

    for i in range(1,num_images-1):
        print(i)
        img2  = cv2.imread(path + '100.png')
        
        # print(cumulative_homographies[i])
        warped_img1 = cv2.warpPerspective(img1, cumulative_homographies[i-1], ((img1.shape[0] + img2.shape[0]), (img1.shape[1])))
        warped_img1[0:img2.shape[0], 0:img2.shape[1]] = img2
        
        img1 = np.array(warped_img1)

    print(img1.shape)
    img1 = img1[:1000,:1000,:]

    cv2.namedWindow('Combined', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Combined', 800, 600)
    cv2.moveWindow('Combined', 1000, 100)
    cv2.imshow('Combined', img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    

    # return np.array(homographies)
