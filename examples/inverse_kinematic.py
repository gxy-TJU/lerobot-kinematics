# code by LinCC111 Boxjod 2025.1.13 Box2AI-Robotics copyright 盒桥智能 版权所有

import os
import mujoco
import mujoco.viewer
import numpy as np
import time
from lerobot_kinematics import lerobot_IK, lerobot_FK, get_robot, feetech_arm

from pynput import keyboard
import threading
import math
# For Feetech Motors
from lerobot_kinematics.lerobot.feetech import FeetechMotorsBus
import json
from Pixel2world import pixel2world

np.set_printoptions(linewidth=200)

# Define joint names
JOINT_NAMES = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]

# Absolute path of the XML model
xml_path = "lerobot-kinematics/examples/scene.xml"
mjmodel = mujoco.MjModel.from_xml_path(xml_path)
qpos_indices = np.array([mjmodel.jnt_qposadr[mjmodel.joint(name).id] for name in JOINT_NAMES])
mjdata = mujoco.MjData(mjmodel)

robot = get_robot('so100')

# Define joint control increment (in radians)
JOINT_INCREMENT = 0.005
POSITION_INSERMENT = 0.0008

# Define joint limits
control_qlimit = [[-2.1, -3.1, -0.0, -1.375,  -1.57, -0.15], 
                  [ 2.1,  0.0,  3.1,  1.475,   3.1,  1.5]]
control_glimit = [[0.125, -0.4,  0.046, -3.1, -0.75, -1.5], 
                  [0.340,  0.4,  0.23, 2.0,  1.57,  1.5]]

# Initialize target joint positions
init_qpos = np.array([0.0, -3.14, 3.14, 0.0, -1.57, -0.157])
target_qpos = init_qpos.copy()
init_gpos = lerobot_FK(init_qpos[1:5],robot=robot)
target_gpos = init_gpos.copy()

init_qpos = np.array([0, -2.443, 2.992, 0.292, -1.570, -0.157])
target_qpos = init_qpos.copy()
init_gpos = lerobot_FK(init_qpos[1:5],robot=robot)
target_gpos = init_gpos.copy()
print("init_gpos", target_gpos)
target_gpos_last = init_gpos.copy()

def move_to_target(world_coord):
    word_coords = world_coord
    step_start = time.time()
    distance = math.sqrt(word_coords[0] * word_coords[0] + word_coords[1] * word_coords[1])
    pitch = 0.842
            
    distance_ = distance - 0.1 * math.sin(pitch)
    distance_err = distance_ - current_gpos[0]
            
    while abs(distance_ - current_gpos[0]) > 0.01:
        print('distance_err', distance_err)
        if distance_err > 0:
            current_gpos[0] = current_gpos[0] + 0.005
        else:
            current_gpos[0] = current_gpos[0] - 0.005
        target_gpos = np.array([current_gpos[0], 0, 0.046, -1.570, pitch, 0])
        print("target_gpos:", [f"{x:.3f}" for x in target_gpos])
        # fd_qpos = np.concatenate(([0.0,], mjdata.qpos[qpos_indices][1:5]))
        fd_qpos = mjdata.qpos[qpos_indices][1:5]
        qpos_inv, IK_success = lerobot_IK(fd_qpos, target_gpos, robot=robot)
        
        if np.all(qpos_inv != -1.0):  # Check if IK solution is valid
            
            angel = np.arctan2(word_coords[1], word_coords[0]) * 1.1
            angel = np.array([angel])
            target_qpos = np.concatenate((angel, qpos_inv[:4], np.array([1])))
            print(target_qpos)
            #print("target_qpos:", [f"{x:.3f}" for x in target_qpos])
            mjdata.qpos[qpos_indices] = target_qpos
            # mjdata.ctrl[qpos_indices] = target_qpos
            
            mujoco.mj_step(mjmodel, mjdata)
            with viewer.lock():
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(mjdata.time % 2)
            viewer.sync()

            follower_arm.action(target_qpos)
            target_gpos_last = target_gpos.copy()
        else:
            target_gpos = target_gpos_last.copy()
            current_gpos[0] = target_gpos[0]
        time.sleep(0.01)
        distance_err = distance_ - current_gpos[0]

# Connect to the robotic arm motors
motors = {"shoulder_pan": (1, "sts3215"),
          "shoulder_lift": (2, "sts3215"),
          "elbow_flex": (3, "sts3215"),
          "wrist_flex": (4, "sts3215"),
          "wrist_roll": (5, "sts3215"),
          "gripper": (6, "sts3215")}

follower_arm = feetech_arm(driver_port="/dev/ttyACM1", calibration_file="/home/gxy/work/idp/lerobot-kinematics/examples/main_follower.json" )
current_gpos = init_gpos.copy()
t = 0
try:
    # Launch MuJoCo viewer
    with mujoco.viewer.launch_passive(mjmodel, mjdata) as viewer:
        start = time.time()
        if t ==0 :
            mjdata.qpos[qpos_indices] = init_qpos
            mujoco.mj_step(mjmodel, mjdata)
            with viewer.lock():
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(mjdata.time % 2)
            viewer.sync()
            mjdata.qpos[qpos_indices] = init_qpos   
            mujoco.mj_step(mjmodel, mjdata)
            follower_arm.action(target_qpos)    
        t = t + 1
        while viewer.is_running():
            #world_coords = pixel2world(320, 240)/100
            move_to_target([0.35, 0, 2])
            
except KeyboardInterrupt:
    print("User interrupted the simulation.")
finally:
    follower_arm.disconnect()
    viewer.close()
