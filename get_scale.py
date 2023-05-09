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
    return

def show_warped_images(warped_img1,img2):
    cv2.namedWindow('Warped', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Warped', 800, 600)
    cv2.moveWindow('Warped', 1000, 100)
    cv2.imshow('Warped', warped_img1)
    # cv2.waitKey(0)

    # Show stitched image
    warped_img1[0:img2.shape[0], 0:img2.shape[1]] = img2
    cv2.namedWindow('Combined', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Combined', 800, 600)
    cv2.moveWindow('Combined', 1000, 100)
    cv2.imshow('Combined', warped_img1)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def get_scale(image_path='poster_dataset/images/0.png', poster_path='poster_dataset/poster_gray.jpg',res=5.67,draw=True):

    # create sift object
    sift = cv2.SIFT_create()
    
    img1 = cv2.imread(image_path)
    img2 = cv2.imread(poster_path)

    # Convert images to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # cv2.imshow('img 1', gray1)
    # cv2.waitKey(0)
    
    # Detect features and compute descriptors
    kp1, desc1 = sift.detectAndCompute(gray1, None)
    kp2, desc2 = sift.detectAndCompute(gray2, None)

    # Uncomment to view keypoints drawn on original images

    # if draw:
    #     draw_keypoints_on_img(img1,img2,kp1,kp2)

    # Match features using brute force
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc1, desc2, k=2)

    # Apply ratio test
    good_matches = []

    # Select one of the two nearest neighbours
    for match in matches:
        if len(match) >= 2:
            m, n = match
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)

    # Uncomment to view matched features
    # # Draw matches
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # if draw:
    #     show_matched_features(img_matches)

    # print('here1')

    # Compute homography matrix
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    # print('here2')
            
    M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 1.0)

    warped_img1 = cv2.warpPerspective(img1, M, ((img2.shape[1]), (img2.shape[0])))
    # # Show warped image only

    # print(warped_img1.shape)
    # print(img2.shape)

    #----------------------       compute scale      ---------------------------------

    h, w = img1.shape[:2]
    corners = np.array([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]], dtype=np.float32)
    wrapped_corners = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), M).reshape(-1, 2)
    width = np.linalg.norm(wrapped_corners[1] - wrapped_corners[0])
    height = np.linalg.norm(wrapped_corners[3] - wrapped_corners[0])
    # print('width, height',width,height)
    real_width = width/res   # width (pixels) / resolution (picels/cm)
    real_height = height/res   # height (pixels) / resolution (picels/cm)
    # print('Real Dimensions',real_width,real_height)

    #---------------------------------------------------------------------------------

    # if draw:
    #     show_warped_images(warped_img1,img2)

    sx = real_width / 710
    sy = real_height / 350

    # S = [[sx,0,0],[0,sy,0],[0,0,1]]
    S = [[sx,0,0],[0,sy,0],[0,0,1]]

    return np.array(S)


# print('S ',get_scale(image_path='poster_dataset/poster_half_vert.jpg'))