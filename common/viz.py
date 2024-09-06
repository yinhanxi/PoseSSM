import os
from common.camera import camera_to_world
import numpy as np
import torch
import cv2
import math
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
plt.switch_backend('agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

def quaternion_to_euler_degrees(rot):  
    """  
    Convert a quaternion from the w, x, y, z format to Euler angles (degrees).  
    The rotation order is Z-Y-X, which is equivalent to yaw, pitch, roll.  
    """  
    [w, x, y, z] = rot
    t0 = +2.0 * (w * x + y * z)  
    t1 = +1.0 - 2.0 * (x * x + y * y)  
    roll_x = math.degrees(math.atan2(t0, t1))  
  
    t2 = +2.0 * (w * y - z * x)  
    # No need to clip for asin in this context, but ensure it's valid input  
    if abs(t2) > 1:  
        t2 = t2 / abs(t2)  # Normalize to avoid math domain error, but this should not happen with valid quaternions  
    pitch_y = math.degrees(math.asin(t2))  
  
    t3 = +2.0 * (w * z + x * y)  
    t4 = +1.0 - 2.0 * (y * y + z * z)  
    yaw_z = math.degrees(math.atan2(t3, t4))  
  
    return yaw_z, pitch_y, roll_x  # In degrees

def show2Dpose(kps, img):
    connections = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5],
                   [5, 6], [0, 7], [7, 8], [8, 9], [9, 10],
                   [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]]

    LR = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0], dtype=bool)

    color = (255, 0, 0)
    thickness = 3

    for j,c in enumerate(connections):
        start = map(int, kps[c[0]])
        end = map(int, kps[c[1]])
        start = list(start)
        end = list(end)
        cv2.line(img, (start[0], start[1]), (end[0], end[1]), color, thickness)
        cv2.circle(img, (start[0], start[1]), thickness=-1, color=color, radius=3)
        cv2.circle(img, (end[0], end[1]), thickness=-1, color=color, radius=3)

    return img

def show3Dpose(pred, gt, ax, azim):
    ax.view_init(elev=15., azim=azim)
    #ax.view_init(elev=yaw_z, azim=pitch_y, roll=roll_x)
    #ax.view_init(elev=0, azim=0, roll=90)

    pcolor=(1,0,0)
    gcolor = (0,0,1)

    I = np.array( [0, 0, 1, 4, 2, 5, 0, 7,  8,  8, 14, 15, 11, 12, 8,  9])
    J = np.array( [1, 4, 2, 5, 3, 6, 7, 8, 14, 11, 15, 16, 12, 13, 9, 10])

    LR = np.array([0, 1, 0, 1, 0, 1, 0, 0, 0,   1,  0,  0,  1,  1, 0, 0], dtype=bool)
        
    for i in np.arange( len(I) ):
        x, y, z = [np.array( [gt[I[i], j], gt[J[i], j]] ) for j in range(3)]
        ax.plot(x, y, z, lw=2, color = gcolor)
        
    for i in np.arange( len(I) ):
        x, y, z = [np.array( [pred[I[i], j], pred[J[i], j]] ) for j in range(3)]
        ax.plot(x, y, z, lw=2, color = pcolor)

    RADIUS = 0.72
    RADIUS_Z = 0.7

    xroot, yroot, zroot = pred[0,0], pred[0,1], pred[0,2]
    ax.set_xlim3d([-RADIUS+xroot, RADIUS+xroot])
    ax.set_ylim3d([-RADIUS+yroot, RADIUS+yroot])
    ax.set_zlim3d([-RADIUS_Z+zroot, RADIUS_Z+zroot])
    ax.set_aspect('auto') # works fine in matplotlib==2.2.2

    white = (1.0, 1.0, 1.0, 0.0)
    ax.xaxis.set_pane_color(white) 
    ax.yaxis.set_pane_color(white)
    ax.zaxis.set_pane_color(white)

    ax.tick_params('x', labelbottom = False)
    ax.tick_params('y', labelleft = False)
    ax.tick_params('z', labelleft = False)

def draw_pose_2D(viz_2D, count, num, keypoints):
    img_2D = np.ones((1000, 1000, 3), np.uint8) * 255
    viz_2D = 500*viz_2D+500
    img_2D = show2Dpose(viz_2D, img_2D)
    if not os.path.exists(f'./img/{keypoints}/2D'):
        os.makedirs(f'./img/{keypoints}/2D')
    cv2.imwrite(f'img/{keypoints}/2D/{count}_{num}.png', img_2D)
    
def draw_pose_3D(viz_3D, viz_3D_gt, rot, count, num, keypoints, azim):
                
    #rot =  [0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088]
    #rot = np.array(rot, dtype='float32')
    #yaw_z, pitch_y, roll_x = quaternion_to_euler_degrees(rot)
    viz_3D = camera_to_world(viz_3D, R=rot, t=0)
    #viz_3D[:, 2] -= np.min(viz_3D[:, 2])
    viz_3D_gt = camera_to_world(viz_3D_gt, R=rot, t=0)
    #viz_3D_gt[:, 2] -= np.min(viz_3D_gt[:, 2])
    fig = plt.figure(figsize=(9.6, 5.4))
    gs = gridspec.GridSpec(1, 1)
    gs.update(wspace=-0.00, hspace=0.05)
    ax = plt.subplot(gs[0], projection='3d')
    show3Dpose(viz_3D, viz_3D_gt, ax, azim)
    if not os.path.exists(f'./img/{keypoints}/3D/{count}/{num}'):
        os.makedirs(f'./img/{keypoints}/3D/{count}/{num}')
    plt.savefig(f'./img/{keypoints}/3D/{count}/{num}/{azim}.png', dpi=200, format='png', bbox_inches = 'tight')
    matplotlib.pyplot.close()