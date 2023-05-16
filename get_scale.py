#!/usr/bin/python3

import numpy as np
import cv2

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
    # warped_img1[0:img1.shape[0], 0:img1.shape[1]] = img1
    # cv2.namedWindow('Combined', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('Combined', 800, 600)
    # cv2.moveWindow('Combined', 1000, 100)
    # cv2.imshow('Combined', warped_img1)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def get_scale(poster_path, image_path ,poster_len,poster_width,draw=True):

    # create sift object
    sift = cv2.SIFT_create()
    
    poster = cv2.imread(poster_path)
    img1 = cv2.imread(image_path)

    # Convert images to grayscale
    poster_gray = cv2.cvtColor(poster, cv2.COLOR_BGR2GRAY)
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    
    r,c = poster_gray.shape 

    sx = poster_len/r
    sy = poster_width/c

    wHp = np.diag([sx,sy,1])

    # print(wHp)
    
    # Detect features and compute descriptors
    kp1, desc1 = sift.detectAndCompute(poster_gray, None)
    kp2, desc2 = sift.detectAndCompute(img1_gray, None)

    # Uncomment to view keypoints drawn on original images

    # if draw:
    #     draw_keypoints_on_img(img1,img2,kp1,kp2)

#     # Match features using brute force
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc1, desc2, k=2)

    # Apply ratio test
    good_matches = []

    # Select one of the two nearest neighbours
    for match in matches:
        if len(match) >= 2:
            m, n = match
            if m.distance < 0.6 * n.distance:
                good_matches.append(m)

    # Uncomment to view matched features
    # # Draw matches
    img_matches = cv2.drawMatches(poster, kp1, img1, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # if draw:
    #     show_matched_features(img_matches)

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
           
    pH1, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 1.0)

    # print(pH1)

    warped_img1 = cv2.warpPerspective(img1, pH1, ((poster.shape[1]), (poster.shape[0])))
    
    # Show warped image only
    if draw:
        show_warped_images(warped_img1, poster)

    return np.array(wHp), np.array(pH1)

    # warped_img1 = cv2.warpPerspective(img1, M, ((poster.shape[1]), (poster.shape[0])))
    # #     # # Show warped image only
    # if draw:
    #     show_warped_images(warped_img1, poster)


# # print('S ',get_scale(image_path='poster_dataset/poster_half_vert.jpg'))