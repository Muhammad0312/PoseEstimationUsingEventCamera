#!/usr/bin/python3
import cv2
import numpy as np

def create_mosaic(ref_image,homographies):

    # Load reference image and homographies
    ref_image = cv2.imread(ref_image)

    # Compute cumulative homographies
    cumulative_homography = np.eye(3)
    cumulative_homographies = [cumulative_homography]
    for H in homographies:
        cumulative_homography = np.dot(H, cumulative_homography)
        cumulative_homographies.append(cumulative_homography)

    # Compute output size and offset
    corners = np.array([[0, 0, 1], [0, ref_image.shape[0], 1], [ref_image.shape[1], ref_image.shape[0], 1], [ref_image.shape[1], 0, 1]])
    corners = np.dot(corners, cumulative_homography.T)
    corners[:, 0] /= corners[:, 2]
    corners[:, 1] /= corners[:, 2]
    min_x = int(np.floor(min(corners[:, 0])))
    max_x = int(np.ceil(max(corners[:, 0])))
    min_y = int(np.floor(min(corners[:, 1])))
    max_y = int(np.ceil(max(corners[:, 1])))
    output_size = (max_x - min_x, max_y - min_y)
    offset = (-min_x, -min_y)

    # Create output image and blend images
    output_image = np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)
    for i, H in enumerate(cumulative_homographies):
        # Warp image
        image = cv2.imread('poster_dataset/images/'+str(i)+'.png')
        warped_image = cv2.warpPerspective(image, H, output_size)
        
        # Add to output image
        if i == 0:
            output_image[int(offset[1]):int(offset[1])+warped_image.shape[0], int(offset[0]):int(offset[0])+warped_image.shape[1], :] = warped_image
        else:
            overlap = (output_image[int(offset[1]):int(offset[1])+warped_image.shape[0], int(offset[0]):int(offset[0])+warped_image.shape[1], :] > 0).astype(np.int32)
            output_image[int(offset[1]):int(offset[1])+warped_image.shape[0], int(offset[0]):int(offset[0])+warped_image.shape[1], :] *= overlap
            warped_image *= overlap
            output_image[int(offset[1]):int(offset[1])+warped_image.shape[0], int(offset[0]):int(offset[0])+warped_image.shape[1], :] += warped_image

    return output_image