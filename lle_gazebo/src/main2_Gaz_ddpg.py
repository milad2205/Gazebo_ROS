#!/usr/bin/env python3

from cProfile import run
import rospy
from std_msgs.msg import Float64
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Vector3
from geometry_msgs.msg import Point
import math
from std_srvs.srv import Empty
from gazebo_msgs.srv import SetModelConfiguration
# -*- coding: utf-8 -*-
from importlib import reload

import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
# import cartpole_pg
# reload(cartpole_pg)

import signal
import sys
import matplotlib.pyplot as plt
import numpy as np
sys.path.append(".")
sys.path.append("../catkin_ws/src/lle_human/lle_gazebo/src/pyDmps")

from pyDmps import dmp_discrete

import LIP_motion
import load_trajectory as ld

reload(dmp_discrete)
reload(LIP_motion)


from ddpg import *
import random
#---------------------------------------------------------
# record = []
def discount_rewards(reward, gamma):
    discounted_ep_rs = np.zeros_like(reward)
    running_add = 0
    for t in reversed(range(0, len(reward))):
        running_add = running_add * gamma + reward[t]
        discounted_ep_rs[t] = running_add
    return discounted_ep_rs

def norm_rewards(reward):
    reward -= np.mean(reward)
    reward /= (np.std(reward) + 0.0001)
    return reward

def cb(s, f):
    print ('recv signal', s)
    sys.exit(0)

action_set = {0:[-10.0,-10],
              1:[10,10],
              2:[10,-10],
              3:[10,0],
              4:[0,0],
              5:[0,10],
              6:[0,-10],
              7:[-10,0],
              8:[-10,10]}
# action_set = {0:[0.0,0.0],
#               1:[3.0, 0.0],
#               2:[-3.0, 0.0]}
action_set2 = {0:[0.0,0.0],
              1:[4.0, 0.0],
              2:[-4.0, 0.0],
              3:[-10.0, -0.0],
              4:[10.0, 0.0]}
action_set1 = {0:[0.3],
              1:[0.4],
              2:[0.5],
              3:[0.6],
              4:[0.7]}

obs, actions, rewards = [], [], []
state_batch, action_batch, reward_batch = [], [], []
batch = 0
eval_param = 0

def reward_add(re, offset):
    l = len(re)
    for i in range(0, l):
        re[i] = re[i] + offset

##############################################
leg = 0
shank_length = 0.39
thigh_length = 0.41
step_number = 0
fall = 0
out_range_left = 0
out_range_right = 0
out_range = 0
leg_length_max = thigh_length + shank_length
stand_leg = "left_leg" or 'right_leg'
my_imu_data_var = Vector3()
my_odomL_data = Point()
my_odomR_data = Point()
my_odomCom_data = Point()

array_e = np.zeros(5000)
array_state = np.zeros(5000)
array_external_force = np.zeros(2000)
array_swing_x = np.zeros(5000)
array_swing_z = np.zeros(5000)
array_state_x = np.zeros(5000) 
array_state_dx = np.zeros(5000) 
array_state_ddx = np.zeros(5000)
array_y_0 = np.zeros(5000)
array_y_1 = np.zeros(5000) 
array_force_0 = np.zeros(5000)
array_force_1 = np.zeros(5000)
array_goal = np.zeros(5000)
array_alpha = np.zeros(5000)
array_betta = np.zeros(5000)

array_theta_hip_l = np.zeros(5000)
array_theta_hip_r = np.zeros(5000)
array_theta_knee_l=np.zeros(5000)
array_theta_knee_r=np.zeros(5000)
array_a=np.zeros(5000)
array_b=np.zeros(5000)
array_c=np.zeros(5000)
array_d=np.zeros(5000)
# ---------------------------------------------------------------------------------
# alg_flag = float(input(' Step Length: \n \
#                 -1:Move backwards with step_length=-0.35cm\n \
#                 0:No steps\n \
#                 1:fix gait 20cm\n \
#                 2:fix gait 35cm\n \
#                 3:fix gait 50cm\n \
#                 4:fix gait 60cm\n '))
                
############################################
if __name__ == "__main__":
    rospy.init_node("listener")

    l_foot_joint_pub = rospy.Publisher('/lle/j_foot_l_position_controller/command', Float64, queue_size=100)
    r_foot_joint_pub = rospy.Publisher('/lle/j_foot_r_position_controller/command', Float64, queue_size=100)
    l_knee_joint_pub = rospy.Publisher('/lle/j_knee_l_position_controller/command', Float64, queue_size=100)
    r_knee_joint_pub = rospy.Publisher('/lle/j_knee_r_position_controller/command', Float64, queue_size=100)
    l_hip_joint_pub = rospy.Publisher('/lle/j_hip_l_position_controller/command', Float64, queue_size=100)
    r_hip_joint_pub = rospy.Publisher('/lle/j_hip_r_position_controller/command', Float64, queue_size=100)
    reset_simulation = rospy.ServiceProxy('/gazebo/reset_world', Empty)
    reset_joints = rospy.ServiceProxy('/gazebo/set_model_configuration', SetModelConfiguration)
    unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
    pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)

    publish_rate = 10000
    rate = rospy.Rate(publish_rate)
    leg = 0
    theta_hip_l=0
    theta_knee_l=0
    theta_hip_r=0
    theta_knee_r=0
    a=0
    b=0
    c=0
    d=0

    i = 0
    # global tau_value
    Tetha = np.array([0 , 0])
    external_force = np.array([0, 0])
    signal.signal(signal.SIGTERM, cb)
    signal.signal(signal.SIGINT, cb)
    trajectory = ld.loadTrajectory("swing_trajectory.txt")
    trajectory_x = trajectory[:, 0]
    trajectory_y = trajectory[:, 1]
    # plt.plot(trajectory_x, trajectory_y, 'r--', lw=2)
    dmp = dmp_discrete.DMPs_discrete(dmps=2, bfs=100)
    dmp.imitate_path(y_des=np.array([trajectory_x, trajectory_y]))
    dmp.goal[0] = 0.2
    gait_time = dmp.goal[0] * 2.2
    if -0.7 < dmp.goal[0] < 0 :
                gait_time = -gait_time
    # tau_value = 1 / gait_time
    dmp.y0[0] = 0
    dmp.reset_state()
    data_point_num = 3
    # tau_value = (1 / dmp.dt)/data_point_num
    # standard_tau = (1 / dmp.dt)/data_point_num
    tau_value = 1/gait_time
    #------------------------------------------------
    # hparams = {
    #         'input_size': 3,
    #         'hidden_size': 10,
    #         'num_actions': 3,
    #         'learning_rate': 0.01
    # }
    # # environment params
    # eparams = {
    #         'num_batches': 40,
    #         'ep_per_batch': 10,
    # }
    # tf.Graph().as_default()
    # sess = tf.Session()
    # agent = cartpole_pg.PolicyGradientAgent(hparams, sess)
    # sess.run(tf.global_variables_initializer())

    env = [3,2] # [state_dim, action_dim]
    agent = DDPG(env)
    observation = np.array([0, 0, 0])
    i_episode = 0
    reward = 0
    done = 0
    # for i in range(200000):
        # fun(lip)

    def imucallback(msg):
        global my_imu_data_var
        # print('llllllllllll', msg.linear_acceleration)
        my_imu_data_var = msg.linear_acceleration #% rospy.get_time()
        return  my_imu_data_var
    def odomLcallback(odom1):
        global my_odomL_data
        # print('llllllllllll', msg.pose.pose)
        my_odomL_data = odom1.pose.pose.position            #% rospy.get_time()
        return  my_odomL_data
    def odomRcallback(odom2):
        global my_odomR_data
        # print('ppppppp', odom2.pose.pose)
        my_odomR_data = odom2.pose.pose.position                      #% rospy.get_time()
        return  my_odomR_data
    def odomComcallback(odom3):
        global my_odomCom_data
        # print('ppppppp', odom2.pose.pose)
        my_odomCom_data = odom3.pose.pose.position                      #% rospy.get_time()
        return  my_odomCom_data
    def listener():
        my_imu_data_var
        my_odomL_data
        my_odomR_data
        my_odomCom_data
        rospy.init_node('listener')
        rospy.Subscriber('/imu' , Imu , imucallback)
        rospy.Subscriber('/position_footL_p3d' , Odometry , odomLcallback)
        rospy.Subscriber('/position_footR_p3d' , Odometry , odomRcallback)
        rospy.Subscriber('/position_Com_p3d' , Odometry , odomComcallback)
        IMU_pub = rospy.Publisher('/my_imu_acc_data', Vector3, queue_size=100)
        ODOML_pub = rospy.Publisher('/my_odomL_pose', Point, queue_size=100)
        ODOMR_pub = rospy.Publisher('/my_odomR_pose', Point, queue_size=100)
        ODOMCOM_pub = rospy.Publisher('/my_odomCom_pose', Point, queue_size=100)
        rate = rospy.Rate(10000)
        IMU_pub.publish(my_imu_data_var)
        ODOML_pub.publish(my_odomL_data)
        ODOMR_pub.publish(my_odomR_data)
        ODOMCOM_pub.publish(my_odomCom_data)
        rate.sleep()
########################################################

        # ddx0 = my_imu_data_var.x  # if ddx = ddx0
        g = 9.806
        zcc = 1.2
        zc = 0.8
        Tc = math.sqrt(zcc/g)
        t = 0.5
        tau = t/Tc
        # x0 = ddx0/(zc/g)
        x0= my_imu_data_var.y  / (g/zcc)
        dx0 = x0 / Tc
        ### states is 1,2,5,6 : x = -Tc*dx
        if x0 < 0 :
            dx0 = -x0 / Tc

        x = x0*math.cosh(tau) + Tc*dx0*math.sinh(tau)
        dx = (x0/Tc)*math.sinh(tau) + dx0*math.cosh(tau)
        ddx = (g/zc) * x
        e = 0.5*dx**2-0.5*(zc/g)*x**2
        # print('x0',x0 ,'dx0', dx0)
#----------------------------------------------------------------------
        global i, external_force, observation, i_episode, agent, state_batch, reward_batch
        global obs, actions, rewards, batch, tau_value, gait_time, leg, reward, done
        global batch_obs, batch_actions, batch_rewards, eval_param, step_number, stand_leg, action_batch
        #S: x dx ddx y[0] y[1]
        #A: dmp.goal[0](-0.7 0.7) tau(0.02,10.0) external_force[0] external_force[1]
        #R: down -10 not down 1
        # action = agent.act(observation,eval_param)
        # next_state = observation
        # agent.perceive(observation, action, reward, next_state, done)
        # observation = next_state
            
        action = agent.noise_action(observation)

        # if i_episode <= 200 :
        #     action = agent.noise_action(observation)
        # if 300 < i_episode <= 400 :
        #     action = agent.noise_action(observation)
        # if 500 < i_episode <= 800 :
        #     action = agent.action(observation)
        # if 700 < i_episode <= 800 :
        #     action = agent.noise_action(observation)
        # if 900 < i_episode <= 1000 :
        #     action = agent.noise_action(observation)
        # if 1100 < i_episode <= 1200 :
        #     action = agent.noise_action(observation)
        
        # if  20000 <= i_episode <= 21000:
        #     action = agent.action(observation) # direct action for test
        # # if org == 1:
        # if i%20 ==0:
        # print(eval_param)
        # action = 3
        Tetha = np.array(action)
        # external_force = np.array(action_set[action])
        y, dy, ddy = dmp.step(tau=tau_value)
        # lip.set_swing_foot_pos(y[0], y[1])
        # y_track, dy_track, ddy_track = dmp.rollout(tau=tau_value )
        # x_foot_swing_ref = y_track[:, 0]
        # z_foot_swing_ref = y_track[:, 1]
        # x_foot_swing_ref = y[0]
        # z_foot_swing_ref = y[1]
        
        if my_odomL_data.y < my_odomR_data.y :
            stand_leg = 'left_leg'
        if my_odomR_data.y < my_odomL_data.y :
            stand_leg = 'right_leg'
###########################################################
        def reset( x0, dx0, z, dt):
            x0 = x0
            dx0 = dx0
            x = x0
            dx = dx0
            z = z
            left_foot_x = 0.0
            left_foot_z = 0.0
            right_foot_x = 0.0
            right_foot_z = 0.0

            left_knee_x = 0.0
            left_knee_z = 0.0
            right_knee_x = 0.0
            right_knee_z = 0.0

            g = 9.806
            ddx = (g / z) * x
            Tc = math.sqrt(z / g)
            # t Will be set to 0,when enter a new state
            t = 0.0
            dt = dt
            stand_leg_world_x = 0.0
            e = 0.0
            capture_point = 0.0
            # stand_leg = stand_leg
            left_leg_length = 0.0
            right_leg_length = 0.0
            f = 0.0
            shank_length = 0.39
            thigh_length = 0.41
            leg_length_max = thigh_length + shank_length

            step_number = 0
            fall = 0
            out_range_left = 0
            out_range_right = 0
            out_range = 0

        def reset_gazebo():
            # ['waist_thighR', 'waist_thighL', 'thighR_shankR', 'thighL_shankL', 'outer_ring_inner_ring', 'inner_ring_boom', 'boom_waist']
            rospy.wait_for_service('gazebo/reset_world')
            try:
                reset_simulation()
            except(rospy.ServiceException) as e:
                print ("reset_world failed!")

            rospy.wait_for_service('gazebo/set_model_configuration')
            try:
                reset_joints("lle", "robot_description", ['j_hip_r', 'j_hip_l', 'j_knee_r', 'j_knee_l',\
                     'j_foot_r', 'j_foot_l', 'human_bar_right_joint', 'human_bar_left_joint'], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                # reset_joints("walker", "robot_description", ['boom_waist', 'outer_ring_inner_ring', 'thighL_shankL', 'thighR_shankR', 'waist_thighL', 'waist_thighR'], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                # robot_state.last_outer_ring_inner_ring_theta = 0.0
            except (rospy.ServiceException) as e:
                print ("reset_joints failed!")

        def update_orbital_energy():
            g = 9.806
            zc = 0.8
            e = (dx**2 - (zc/g) * x**2) / 2
        # e = 0.5*dx**2-0.5*(zc/g)*x**2
        def update():
            global step_number, fall, out_range_left, out_range_right
            # step_number = 0
            step_number = step_number + 1
            t = 0.5 
            dt = 0.001
            t = t + dt
            # update_motion_state()
            # update_motion_state_recur(f)
            # update_knee_pos()
            update_orbital_energy()
            # update_leg_Length()

            observation = [x, dx, ddx]
            # print(observation)
            reward = 0.0
            done = 0
            out_range_left   = 0
            out_range_right = 0
            fall = 0
            # if my_odomL_data.y  < my_odomCom_data.y < my_odomR_data.y  or \
            #     my_odomR_data.y  < my_odomCom_data.y < my_odomL_data.y  :
            #     fall = 0
            # else :
            #     fall = 1

            if stand_leg == 'left_leg' and my_odomL_data.y - my_odomCom_data.y >= 0.12 :
                fall = 1
            if stand_leg == 'right_leg' and my_odomR_data.y - my_odomCom_data.y >= 0.12  :
                fall = 1
            if stand_leg == 'left_leg' and my_odomCom_data.y - my_odomR_data.y >= 0.15:
                fall = 1
            if stand_leg == 'right_leg' and my_odomCom_data.y - my_odomL_data.y >= 0.15 :
                fall = 1

            if my_imu_data_var.x >2 or my_imu_data_var.y >3 or my_imu_data_var.z < 5 :
                fall = 1
            
            if my_odomCom_data.x > 2 or my_odomCom_data.x < -2 or my_odomCom_data.y <= -0.15 :
                fall = 1


            out_range = out_range_left or out_range_right

            # if step_number >= 5000:
            #     done = 1
            if fall == 1:
                reward = -2.0
                done = 2

            info = step_number
            # print('done', done, 'step_number', step_number,'fail',fall,'out_range',out_range)
            return np.array(observation), reward, done, info
#########################################################
        observation, reward, done, info = update()
        global theta_hip_l,theta_knee_l,theta_hip_r,theta_knee_r,a,b,c,d
        array_e[i] = e
        # array_a[i] = a
        # array_b[i] = b
        # array_c[i] = c
        # array_d[i] = d
        # array_theta_hip_l[i] = theta_hip_l
        # array_theta_hip_r[i] = theta_hip_r
        # array_theta_knee_l[i]=theta_knee_l
        # array_theta_knee_r[i]=theta_knee_r
        # array_swing_x[i] = lip.stand_leg_world_x + lip.left_foot_x
        # array_swing_z[i] = lip.left_foot_z
        array_swing_x[i] =  my_odomL_data.y
        array_swing_z[i] = my_odomL_data.z

        array_state_x[i] = x
        array_state_dx[i] = dx
        array_state_ddx[i] = ddx
        array_y_0[i] = y[0]
        array_y_1[i] = y[1]
        array_force_0[i] = external_force[0]
        array_force_1[i] = external_force[1]
        array_goal[i] = dmp.goal[0]
        # array_alpha[i] = alpha_
        # array_betta[i] = betta
        # print('swing' ,z_foot_swing_ref[1])
        if y[1] <= 0.0:
            # global alpha, betta
            update_orbital_energy()
            # reward1 = np.log(1.0/(x + 0.01)**2 +0.001)
            # reward2 = np.log(1.0/((e - 0.1)**2 +0.001))
            # reward = reward1 + reward2
            # reward = np.log(1.0/((e - 0.065)**2+0.001))
            reward = np.log(1.0/((e - 0.2)**2+0.001))
            # reward = (-dmp.y0[0])*20.0 + lip.stand_leg_world_x*10.0
            # print('E',lip.E)
            # print('rewards',reward,'len',(-dmp.y0[0])*10.0)
            # dmp.goal[0] = 0.3
            alpha_ = Tetha[0]
            betta = Tetha[1]
            array_alpha[i] = alpha_
            array_betta[i] = betta
            X_com = random.uniform(0.05,0.15)
            # X_com = 0.15
            # print('X_com' , X_com)
            X_leg = X_com * 2.1
            # X_leg = random.uniform(0.2,0.4)
            g_init = 0.2
            dmp.goal[0] = g_init + alpha_* ( betta- (X_com/X_leg))

            gait_time = dmp.goal[0] * 2.2
            if -5 < dmp.goal[0] < 0 :
                gait_time = -gait_time
            tau_value = 1 / gait_time
            dmp.reset_state()
            print('\nx:',x ,'dx:',dx ,'E:',e)
            print('action=',action,'goal=',dmp.goal[0])
            print('X_com' , X_com,'tau',tau_value )
        i = i + 1
        obs.append(observation)
        actions.append(action)
        rewards.append(reward)
        
        if done :
            global running_reward, i_episode
            batch = batch + 1
            dmp.goal[0] = 0.2
            gait_time = dmp.goal[0] * 2.2
            if -0.7 < dmp.goal[0] < 0 :
                gait_time = -gait_time
            tau_value = 1 / gait_time
            dmp.y0[0] = 0.0
            dmp.reset_state()
            # reward_add(rewards, lip.stand_leg_world_x * 2)
            # reward_add(rewards, offset = e**2)
            # reward_add(rewards, offset = my_odomCom_data.y *2 )
            running_reward = np.sum(rewards)
            # print('X', lip.stand_leg_world_x*100.0)
            reset_gazebo()
            reset(x0, dx0, 0.8, 0.001)
            i_episode = i_episode + 1
            # if eval_param == 1:
            print("episode:", i_episode, "rewards:", running_reward)
            # if eval_param == 1:
            #     eval_param = 0
            rewards = discount_rewards(rewards, 0.95)
            action_batch.extend(actions)
            state_batch.extend(obs)
            reward_batch.extend(rewards)
            obs, actions, rewards = [], [], []

            if batch == 1:
                # global theta_hip_l,theta_knee_l,theta_hip_r,theta_knee_r,a,b,c,d
                fig = plt.figure(1)
                # ax = fig.add_subplot(111)
                plt.plot(reward_batch)
                axx = plt.gca()
                axx.invert_yaxis()
                plt.xlabel('Reward')
                plt.grid(True)
                # if i_episode == 1:
                #     plt.show()

                plt.figure(2)
                plt.plot(array_e)
                plt.title('Energy')
                plt.ylabel('E')
                # plt.grid(True)
                # plt.show()

                plt.figure(3)
                plt.plot(array_state_x, 'r')
                plt.plot(array_state_dx, 'b')
                plt.plot(array_state_ddx, 'green')
                plt.plot(array_y_0, 'orange')
                plt.plot(array_y_1, 'black')
                # plt.plot(array_y_0, 'orange')
                # plt.plot(array_y_1, 'black')
                plt.legend(["x" , "dx" , 'ddx', 'y[0]', 'y[1]'],loc='upper right')
                plt.title('Observation')
                plt.ylabel('Theree_State')
                # plt.grid(True)
                # plt.show()

                # plt.figure(4)
                # plt.plot(array_swing_x, array_swing_z)
                # plt.title('Foot_Trajectory')
                # plt.xlabel('swing_x')
                # plt.ylabel('swing_z')
                # plt.grid(True)
                # plt.show()
                plt.figure(4)
                plt.plot(array_swing_x, array_swing_z)
                plt.title('Foot_Trajectory')
                plt.xlabel('swing_x')
                plt.ylabel('swing_z')             

                plt.figure(7)
                plt.plot(array_alpha, 'r')
                plt.plot(array_betta, 'b')
                plt.plot(array_goal, 'green')
                plt.legend(["alpha" , "betta" , 'goal'],loc='upper right')
                plt.title('Step_Length')
                plt.ylabel('dmp.goal')
                if running_reward >= 5000:
                    plt.show()
    
                #     fig.show()
                # if i_episode == 700:
                #     plt.show()
                #     fig.show()
                # if i_episode == 800:
                #     plt.show()
                #     fig.show()
                    
                # if i_episode == 1100:
                #     plt.show()
                #     fig.show()
                if i_episode == 1200:
                    # fig.show()
                    plt.show()
                    
                # fig.show()
                # plt.plot(np.arange(len(batch_rewards)), batch_rewards)
                # plt.show()
                # plt.show(block=False)
                # plt.pause(0.000001)
       
            #--------------------------------------------------------------
                # sys.exit()
                state_batch, action_batch, reward_batch = [], [], []
                batch = 0
                # eval_param = 1
            i = 0
    #################################################################
        # y_track, dy_track, ddy_track = dmp.rollout(tau=tau_value )

        def com_tarj(x0, dx0, z0, t):
                x0 = x0
                dx0 = dx0
                z = z0
                t = t
                g = 9.806
                Tc = math.sqrt(g/z)
                tau = (t / Tc)
                x = x0*math.cosh(tau) + Tc*dx0*math.sinh(tau)
                dx = (x0/Tc)*math.sinh(tau) + dx0*math.cosh(tau)
                ddx = (g / z) * x

                return x, dx, z 

        dt = 0.001
        # gait_time = 1
        final_t = gait_time
        t = np.arange(0, final_t, dt)
        ts = int(final_t / dt)
        x0 = 0.1
        dx0 = 0.1
        z0 = 0.8

        x = np.zeros(t.shape)
        dx = np.zeros(t.shape)
        ddx = np.zeros(t.shape)
        z = np.zeros(t.shape)

        for i in range(ts):
            x[i],dx[i], z[i] = com_tarj(x0, dx0, z0, t[i])
        
    #####################################
        y_track, dy_track, ddy_track = dmp.rollout(tau=tau_value)
        x_foot_swing_ref = y_track[:, 0]
        z_foot_swing_ref = y_track[:, 1]
        # global z_foot_swing_ref

        x_com_ref = x
        z_com_ref = z

        
        theta_knee_l = np.zeros(t.shape)
        theta_hip_l = np.zeros(t.shape)
        theta_knee_r = np.zeros(t.shape)
        theta_hip_r = np.zeros(t.shape)

        x_r = np.zeros(t.shape)
        z_r = np.zeros(t.shape)
        x_l = np.zeros(t.shape)
        z_l = np.zeros(t.shape)

        a = np.zeros(t.shape)
        b = np.zeros(t.shape)
        c = np.zeros(t.shape)
        d = np.zeros(t.shape)

        q = np.zeros(t.shape)
        w = np.zeros(t.shape)
        e = np.zeros(t.shape)
        u = np.zeros(t.shape)

        for tt in range(ts):
            # if stance_leg == "left_leg":

                x_foot_left = 0
                z_foot_left = 0

                x_r[tt] = x_foot_swing_ref[tt] - x_com_ref[tt]
                z_r[tt] = z_foot_swing_ref[tt] - z_com_ref[tt]

                x_l[tt] = x_foot_left - x_com_ref[tt]
                z_l[tt] = z_foot_left - z_com_ref[tt]

                theta_knee_l[tt] =math.degrees(np.arccos((shank_length**2 + thigh_length**2 - x_l[tt]**2 - z_l[tt]**2)/(2*shank_length*thigh_length))) - 180 
                theta_hip_l[tt] = math.degrees(np.arctan((-x_l[tt]/z_l[tt])) + np.arccos((x_l[tt]**2 + z_l[tt]**2 + thigh_length**2 - shank_length**2)/(2*thigh_length*np.sqrt(x_l[tt]**2 + z_l[tt]**2))))
                
                theta_knee_r[tt] = math.degrees(np.arccos((shank_length**2 + thigh_length**2 - x_r[tt]**2 - z_r[tt]**2)/(2*shank_length*thigh_length))) - 180
                theta_hip_r[tt] = math.degrees(np.arctan((-x_r[tt]/z_r[tt])) + np.arccos((x_r[tt]**2 + z_r[tt]**2 + thigh_length**2 - shank_length**2)/(2*thigh_length*np.sqrt(x_r[tt]**2 + z_r[tt]**2))))
                
            # if stance_leg == "left_leg":
                x_foot_right = 0
                z_foot_right = 0

                q[tt] = x_foot_swing_ref[tt] - x_com_ref[tt]
                w[tt] = z_foot_swing_ref[tt] - z_com_ref[tt]

                e[tt] = x_foot_right - x_com_ref[tt]
                u[tt] = z_foot_right - z_com_ref[tt]

                a[tt] =math.degrees(np.arccos((shank_length**2 + thigh_length**2 - q[tt]**2 - w[tt]**2)/(2*shank_length*thigh_length))) - 180 
                b[tt] = math.degrees(np.arctan((-q[tt]/w[tt])) + np.arccos((q[tt]**2 + w[tt]**2 + thigh_length**2 - shank_length**2)/(2*thigh_length*np.sqrt(q[tt]**2 + w[tt]**2))))
                
                c[tt] = math.degrees(np.arccos((shank_length**2 + thigh_length**2 - e[tt]**2 - u[tt]**2)/(2*shank_length*thigh_length))) - 180
                d[tt] = math.degrees(np.arctan((-e[tt]/u[tt])) + np.arccos((e[tt]**2 + u[tt]**2 + thigh_length**2 - shank_length**2)/(2*thigh_length*np.sqrt(e[tt]**2 + u[tt]**2))))
                
        leg = leg +1       
        # if stand_leg == 'left_leg':
        if leg == 1 :
            t=0
            while t <ts :
                l_knee_joint_pub.publish(math.radians(theta_knee_l[t]))
                r_knee_joint_pub.publish(math.radians(theta_knee_r[t]))
                l_hip_joint_pub.publish(math.radians(theta_hip_l[t]))
                r_hip_joint_pub.publish(math.radians(theta_hip_r[t]))
                t = t +1 
                rate.sleep()
                   
        # leg = leg +1
        # if stand_leg == 'right_leg':
        if leg == 2:
            t=0
            while t <ts :
                l_knee_joint_pub.publish(math.radians(a[t]))
                r_knee_joint_pub.publish(math.radians(c[t]))
                l_hip_joint_pub.publish(math.radians(b[t]))
                r_hip_joint_pub.publish(math.radians(d[t]))
                t = t +1 
                rate.sleep()
            leg = 0  
###########################################################
    while not rospy.is_shutdown():

        listener()
        