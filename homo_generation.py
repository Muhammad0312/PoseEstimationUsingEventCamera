#!/usr/bin/python3

import numpy as np
import cv2

from get_scale import *

def draw_keypoints_on_img(img1,img2,kp1,kp2):
    img_kp = cv2.drawKeypoints(img1, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    img_kp1 = cv2.drawKeypoints(img2, kp2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Display the output image1
    cv2.namedWindow('SIFT Keypoints', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('SIFT Keypoints', 800, 600)
    cv2.imshow('SIFT Keypoints', img_kp)
    # Display the output image2
    cv2.namedWindow('SIFT Keypoints1', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('SIFT Keypoints1', 800, 600)
    cv2.imshow('SIFT Keypoints1', img_kp1)
    cv2.waitKey(0)

def show_matched_features(img_matches):
    cv2.namedWindow('Matched Features', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Matched Features', 800, 600)
    cv2.moveWindow('Matched Features', 100, 100)
    # Display result
    cv2.imshow('Matched Features', img_matches)
    cv2.waitKey(0)

def show_warped_images(warped_img1,img2):
    cv2.namedWindow('Warped', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Warped', 800, 600)
    cv2.moveWindow('Warped', 1000, 100)
    cv2.imshow('Warped', warped_img1)
    cv2.waitKey(0)

    # Show stitched image
    warped_img1[0:img2.shape[0], 0:img2.shape[1]] = img2
    cv2.namedWindow('Combined', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Combined', 800, 600)
    cv2.moveWindow('Combined', 1000, 100)
    cv2.imshow('Combined', warped_img1)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def get_homography(dataset,path,num_images,draw=False,scale=False):

    # list to append all image paths
    imagePaths = []
    # list to append all homographies
    all_homographies = []

    # Open the file for reading
    with open(path, 'r') as file:
        # Read each line of the file into a list
        lines = file.readlines()
        # Read each line separately
        for line in lines:
            words = line.split()
            imagePaths.append(words[-1])


    # create sift object
    sift = cv2.SIFT_create()
    jump = 1
    for i in range(0, num_images,jump):

        try:
            # Read images
            img1 = cv2.imread(dataset+'/'+imagePaths[i])
            img2 = cv2.imread(dataset+'/'+imagePaths[i +  jump])

            # Convert images to grayscale
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

            # cv2.imshow('img 1', gray1)
            # cv2.waitKey(0)
            
            # Detect features and compute descriptors
            kp1, desc1 = sift.detectAndCompute(gray1, None)
            kp2, desc2 = sift.detectAndCompute(gray2, None)

            # Uncomment to view keypoints drawn on original images

            if draw:
                draw_keypoints_on_img(img1,img2,kp1,kp2)

            # Match features using brute force
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(desc1, desc2, k=2)

            # Apply ratio test
            good_matches = []

            # Select one of the two nearest neighbours
            for match in matches:
                if len(match) >= 2:
                    m, n = match
                    if m.distance < 0.9 * n.distance:
                        good_matches.append(m)

            # Uncomment to view matched features
            # # Draw matches
            img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

            if draw:
                show_matched_features(img_matches)

            # Compute homography matrix
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            
            if len(src_pts) >= 4 and len(dst_pts) >= 4:               
                M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 1.0)

                if scale:
                    # give rescaling the destination image
                    S = get_scale(image_path=dataset+'/'+imagePaths[i +  jump])
                    # scale the homogrpahy matrix
                    M = S @ M @ S.T

                all_homographies.append(np.array(M))

                warped_img1 = cv2.warpPerspective(img1, M, ((img1.shape[0] + img2.shape[0]), (img1.shape[1])))
                # # Show warped image only
                if draw:
                    show_warped_images(warped_img1,img2)
                
            else:
                print('Bro Give me more points')
            
        except:
            print('Frame Skipped')
            print(imagePaths[i])
            print(imagePaths[i +  1])

    # print(all_homographies)
    

    # with open(dataset+'/image_homographies.txt', 'w') as f:
    #     f.write('[')
    #     for item in range(0, len(all_homographies)):
    #         f.write("%s" % all_homographies[item])
    #         if item != len(all_homographies) - 1:
    #             f.write(",\n")
    #     f.write(']')

    #     height, width = img2.shape[:2]
    #     warped_img1 = cv2.warpPerspective(img1, M, (width, height))

    #     panorama = cv2.addWeighted(warped_img1, 0.5, img2, 0.5, 0)

        # Display result
        # cv2.namedWindow('Panorama', cv2.WINDOW_NORMAL)

        # # Set window size
        # cv2.resizeWindow('Panorama', 800, 600)

        # cv2.imshow('Panorama', panorama)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    return np.array(all_homographies)
