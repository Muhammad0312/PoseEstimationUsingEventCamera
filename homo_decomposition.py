#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import cv2
import matplotlib.cm as cm

def get_transform_from_homography(H, K, prev_T=None):
    H = H / H[2, 2] #The reason for doing this is to make the homography matrix invariant to scale.
    _, R_array, t_array, _ = cv2.decomposeHomographyMat(H, K)

    closest_R = None
    closest_diff = float('inf')
    if len(t_array)==1:
      R=R_array[0]
      t=t_array
    #   print ("t_array",t_array)
    #   print ("R_array",R_array)
    else:
      for i in range(4):
          t = t_array[i]
          if t[2] > 0:
              R = R_array[i]
              ###### closest orientation to the previous frame #####
              if prev_T is not None:
                  prev_R = prev_T[:3, :3]
                  diff = np.linalg.norm(R - prev_R)
                  if diff < closest_diff:
                      closest_R = R
                      closest_diff = diff
              else:
                  closest_R = R
                  closest_diff = 0
              ######################################################
   
    if closest_R is None: #just for a bug .. incase it returns None
        closest_R = R_array[0]
        closest_diff = 0
    C = -closest_R.T @ t
    T = np.eye(4)
    T[:3, :3] = R.T
    T[:3, 3] = C.flatten()
    return T
    
def get_camera_trajectory(H_list,K):
    camera_traj = np.zeros((3, 1))
    x_pos, y_pos,z_pos = [0], [0],[0]
    camera_pose= np.eye(4)
    for H in H_list:
        # trans_mat = get_transform_from_homography(np.array(H),K)
        trans_mat = get_transform_from_homography(np.array(H),K,camera_pose)
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
    colors = np.linspace(0, 1, len(x_pos))
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.scatter(x_pos, y_pos, c=colors)
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    # ax.set_xlim(-2, 10)
    # ax.set_ylim(-2, 10)
    ax.set_title('Robot Trajectory')
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
    ax.set_title('Camera Trajectory')
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
    ax.set_title('Camera Trajectory')
    ax.legend()
    plt.show()

