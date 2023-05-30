#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import cv2
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D

# def get_transform_from_homography(H, K, prev_T=None):
#     H = H / H[2, 2] #The reason for doing this is to make the homography matrix invariant to scale.
#     _, R_array, t_array, _ = cv2.decomposeHomographyMat(H, K)

#     closest_R = None
#     closest_diff = float('inf')
#     if len(t_array)==1:
#       R=R_array[0]
#       t=t_array
#     #   print ("t_array",t_array)
#     #   print ("R_array",R_array)
#     else:
#       for i in range(4):
#           t = t_array[i]
#           if t[2] > 0:
#               R = R_array[i]
#               ###### closest orientation to the previous frame #####
#               if prev_T is not None:
#                   prev_R = prev_T[:3, :3]
#                   diff = np.linalg.norm(R - prev_R)
#                   if diff > closest_diff:
#                       closest_R = R
#                       closest_diff = diff
#               else:
#                   closest_R = R
#                   closest_diff = 0
#               ######################################################
   
#     if closest_R is None: #just for a bug .. incase it returns None
#         closest_R = R_array[0]
#         closest_diff = 0
#     C = -closest_R.T @ t
#     T = np.eye(4)
#     T[:3, :3] = R.T
#     T[:3, 3] = C.flatten()
#     # T = np.eye(4)
#     # T[:3, :3] = closest_R
#     # # print('----------')
#     # T[:3, 3] = np.array(t[0]).flatten()
#     return T
    
# # def get_transform_from_homography(H, K):
# #     # Normalize homography matrix
# #     H = H / H[2, 2]
# #     # Decompose homography into rotation and translation
# #     _, R_array, t_array, _ = cv2.decomposeHomographyMat(H, K)

# #     # Determine camera position
# #     closest_R = None
# #     closest_diff = float('inf')
# #     if len(t_array)==1:
# #       R=R_array[0]
# #       t=t_array
# #     #   print ("t_array",t_array)
# #     #   print ("R_array",R_array)
# #     else:
# #         for i in range(4):
# #             t = t_array[i]
# #             if t[2] > 0:
# #                 R = R_array[i]
# #                 if closest_R is not None:
# #                     # Find the rotation matrix with smallest angle to the current closest_R
# #                     diff_R = R @ closest_R.T
# #                     trace = np.trace(diff_R)
# #                     angle = np.arccos((trace - 1) / 2)
# #                     if angle < closest_diff:
# #                         closest_R = R
# #                         closest_diff = angle
# #                 else:
# #                     closest_R = R
# #                     closest_diff = 0
# #     # If no valid solution found, use the first solution
# #     if closest_R is None:
# #         closest_R = R_array[0]
# #         closest_diff = 0


# #     T = np.eye(4)
# #     T[:3, :3] = closest_R
# #     # print('----------')
# #     T[:3, 3] = np.array(t[0]).flatten()

# #     print(f"Chose solution with smallest angle {closest_diff}")
# #     return T

# # def get_transform_from_homography(H, K):
# #     # Normalize homography matrix
# #     H = H / H[2, 2] #The reason for doing this is to make the homography matrix invariant to scale.
# #     # Decompose homography into rotation and translation
# #     _, R_array, t_array, _ = cv2.decomposeHomographyMat(H, K)

# #     """  checks whether the z-coordinate of the current vector t[2] is positive. If it is, then the corresponding 
# #      rotation matrix R is assigned to the variable R. 
# #      This means that the loop will only consider translation vectors that correspond to a camera position in front of the scene"""
# #     if len(t_array)==1:
# #       R=R_array[0]
# #       t=t_array
# #       print ("t_array",t_array)
# #       print ("R_array",R_array)
# #     else:

# #       # Determine camera position
# #       for i in range(4):
# #           t = t_array[i]
# #           if t[2] > 0:
# #               R = R_array[i]
# #               break
# #       #the camera position C is computed by multiplying the transpose of the rotation matrix R.T with the translation vector t      
# #     # C = -R.T @ t
# #     # Construct 4x4 transformation matrix
# #     T = np.eye(4)
# #     T[:3, :3] = R
# #     # print('----------')
# #     T[:3, 3] = np.array(t[0]).flatten()
# #     # print (T)
# #     return T

def get_camera_trajectory(H_list,K):
    camera_traj = np.zeros((3, 1))
    x_pos, y_pos,z_pos = [0], [0],[0]
    camera_pose= np.array([[1,0,0,0.397],[0,1,0,0.537],[0,0,1,0],[0,0,0,1]])
    for H in H_list:
        # trans_mat = get_transform_from_homography(np.array(H),K)
        trans_mat = get_transform_from_homography(np.array(H),K)#,camera_pose)
        # print ("trans_mat",trans_mat)
        # trans_mat=np.linalg. inv(trans_mat) 
        camera_pose= camera_pose @ trans_mat
        # print ("camera_pose",camera_pose)
        camera_traj = np.dot(camera_pose, np.array([0, 0, 0, 1]).reshape((4, 1)))
        x_pos.append(float(camera_traj[0]))
        y_pos.append(float(camera_traj[1]))
        z_pos.append(float(camera_traj[2]))
        # print (camera_traj)
    return x_pos, y_pos,z_pos


def plot_camera_trajectory(homography_list, K):
    # x,y camera trajectory
    x_pos, y_pos, z_pos = get_camera_trajectory(homography_list, K)
    x_pos, y_pos, z_pos=x_pos[1:], y_pos[1:], z_pos[1:]
    print(x_pos)
    scale_x= 0.397/x_pos[1]
    scale_y= 0.537/y_pos[1]
    x_pos= np.array(x_pos) *scale_x
    y_pos=np.array(y_pos) *scale_y

    colors = np.linspace(0, 1, len(x_pos))
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.scatter(x_pos, y_pos, c=colors)
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    # ax.set_xlim(-2, 10)
    # ax.set_ylim(-2, 10
    ax.set_title('Camera Trajectory |Intensity image- DLT')
    plt.show()

 
    T = np.arange(len(x_pos))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax.set_xlim(-2, 10)
    # ax.set_ylim(-2, 10)
    # ax.set_zlim(-2, 10)
    ax.scatter(x_pos, y_pos, z_pos, c=colors)
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')
    ax.set_title('Camera Trajectory |Intensity image- DLT')
    plt.show()

    fig, ax = plt.subplots()
    # T = np.arange(len(data[:, 0]))
    # plt.plot(T, data[:, 1].T, label='X')
    # plt.plot(T, data[:, 2].T, label='Y')
    # plt.xlabel('Time')
    # plt.ylabel('Position')
    # plt.legend()

    ax.scatter(np.arange(len(x_pos)), x_pos, c=np.linspace(0, 0.3, len(x_pos)), s=2, label='X position')
    ax.scatter(np.arange(len(y_pos)), y_pos,color='Lime', s=2, label='Y position')
    ax.set_xlabel('Time')
    # ax.set_xlim(0,60)
    # ax.set_ylim(-1, 1)
    ax.set_ylabel('Position')
    ax.set_title('Camera Trajectory |Intensity image- DLT')
    ax.legend()
    plt.show()

def get_transform_from_homography(H, K):
    """
    :Target: Compute the camera pose from a homography and cam matrix.
    :param H: the homography matrix.(numpy.ndarray) (3x3)
    :param K: the camera matrix. ( numpy.ndarray)
    :returns: the camera position and orientation as a 3x4 matrix.( numpy.ndarray)
    """
    # mathematical derivation for Decomposition explained here: https://shorturl.at/blnS0

    # Convert homogeneous coordinates to non-homogeneous coordinates
    s = np.sqrt(H[0, 0] * H[1, 1])
    H = H / s
    
    # Extract the rotation and translation matrices from the homography matrix
    # H = K * [R | t]
    h1, h2, h3 = H[:, 0], H[:, 1], H[:, 2]  # the direction of the camera axes in 3D space
    invK = np.linalg.inv(K)
    lamda = 1 / np.linalg.norm(np.dot(invK, h1))
    r1 = lamda * np.dot(invK, h1)
    r2 = lamda * np.dot(invK, h2)
    r3 = np.cross(r1, r2)
    t = lamda * (invK @ h3.reshape(3, 1))

    # Compute the rotation matrix
    R = np.array([[r1], [r2], [r3]])
    R = np.reshape(R, (3, 3)) 

    # Return the camera position and orientation as a 3x4 matrix
    # print(np.hstack((R, t.reshape(3, 1))), -np.dot(R.T, t))

    T = np.eye(4)
    T[:3, :3] = R.T
    # print('----------')
    T[:3, 3] = np.array(-np.dot(R.T, t)).flatten()
    # print (T)
    return T

    # return np.hstack((R, t.reshape(3, 1))), -np.dot(R.T, t)