#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import cv2
import matplotlib.cm as cm



def get_transform_from_homography(H, K):
    # Normalize homography matrix
    H = H / H[2, 2] #The reason for doing this is to make the homography matrix invariant to scale.
    # print('new H:', H)
    # Decompose homography into rotation and translation
    _, R_array, t_array, _ = cv2.decomposeHomographyMat(H, K)

    """  checks whether the z-coordinate of the current vector t[2] is positive. If it is, then the corresponding 
     rotation matrix R is assigned to the variable R. 
     This means that the loop will only consider translation vectors that correspond to a camera position in front of the scene"""
    
    print('Rotation Matrices', R_array)
    print('Translation Matrices', t_array)
    # Determine camera position
    if len(t_array) == 4:
        # print('t array: ', t_array)
        for i in range(4):
            t = t_array[i]
            if t[2] > 0:
                R = R_array[i]
                break
        #the camera position C is computed by multiplying the transpose of the rotation matrix R.T with the translation vector t      
        C = -R.T @ t
        # Construct 4x4 transformation matrix
        T = np.eye(4)
        T[:3, :3] = R.T
        T[:3, 3] = C.flatten()
        # print (T)
        return T
    else:
        return np.eye(4)
    
def get_camera_trajectory(H_list,K):
    camera_traj = np.zeros((3, 1))
    x_pos, y_pos = [0], [0]
    R = np.eye(3)
    T = np.zeros((3,))

    for H in H_list:
        # print(H)
        trans_mat = get_transform_from_homography(np.array(H),K)
        R = np.dot(R, trans_mat[:3, :3])
        T = T + trans_mat[:3, 3]
        camera_pos = np.eye(4)
        camera_pos[:3, :3] = R
        camera_pos[:3, 3] = T
        camera_traj = np.dot(camera_pos, np.array([0, 0, 0, 1]).reshape((4, 1)))
        x_pos.append(float(camera_traj[0]))
        y_pos.append(float(camera_traj[1]))

    return tuple([0] + x_pos), tuple([0] + y_pos)

def plot_camera_trajectory(homography_list,K):
    """
    This function takes in a list of homography matrices and plots the camera trajectory in 2D space
    and 3D space with respect to time.
    
    Args:
    homography_list: A list of homography matrices
    
    Returns:
    None
    """
    # Call the function to get the camera trajectory
    x_pos, y_pos = get_camera_trajectory(homography_list,K)

    # Create a color map with the same length as x_pos and y_pos
    colors = np.linspace(0, 1, len(x_pos))

    # Plot the points in 2D space
    fig, ax = plt.subplots()
    ax.scatter(x_pos, y_pos, c=colors)
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title('Robot Trajectory')
    ax.set_ylim(-0.1, 0.3)
    plt.show()

    # Plot the points in 3D space with respect to time
    X_list = []
    Y_list = []
    Z_list = []

    # initialize the starting position
    x = 0
    y = 0
    z = 0

    # loop through each transformation matrix and compute the new position
    for H in homography_list:
        # extract the translation components of the matrix
        tx = H[0, 2]
        ty = H[1, 2]
        tz = H[2, 2]

        # update the current position with the translation
        x += tx
        y += ty
        z += tz

        # append the current position to the X_list, Y_list, and Z_list
        X_list.append(x)
        Y_list.append(y)
        Z_list.append(z)

    # Create a color map with the same length as X_list, Y_list, and Z_list
    colors = np.linspace(0, 1, len(X_list))

    # plot the X, Y, and Z positions with respect to time
    T = np.arange(len(X_list))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_list, Y_list, Z_list, c=colors)
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')
    ax.set_title('Robot Trajectory')
    plt.show()


    # plot the X and Y positions with respect to time
    fig, ax = plt.subplots()
    ax.plot(np.arange(len(X_list)), X_list, label='X position')
    ax.plot(np.arange(len(Y_list)), Y_list, label='Y position')
    ax.set_xlabel('Time')
    ax.set_ylabel('Position')
    ax.set_title('X, Y wrt Time')
    ax.legend()
    plt.show()