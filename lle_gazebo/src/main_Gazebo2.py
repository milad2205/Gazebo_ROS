
import rospy
from std_msgs.msg import Float64
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Vector3
import math
from std_srvs.srv import Empty
from gazebo_msgs.srv import SetModelConfiguration
# -*- coding: utf-8 -*-
from importlib import reload

import signal
import sys
import matplotlib.pyplot as plt
import numpy as np
sys.path.append("./pyDmps")
sys.path.append(".")
import dmp_discrete
import LIP
import LIP_motion
import load_trajectory as ld
import circular_inter_fast
import net
reload(net)
# reload(LIP)
reload(dmp_discrete)
reload(LIP_motion)
# reload(circular_inter_fast)
#---------------------------------------------------------
def cb(s, f):
    print ('recv signal', s)
    RL.sess.close()
    sys.exit(0)
# action_set = {0:[-10.0,-10],
            #    1:[10,10],
            #    2:[10,-10],
            #    3:[10,0],
            #    4:[0,0],
            #    5:[0,10],
            #    6:[0,-10],
            #    7:[-10,0],
            #    8:[-10,10]}
action_set = {0:[-5.0,-5.0],
            1:[5.0,5.0],
            2:[5.0,-5.0],
            3:[5.0,0.0],
            4:[0, 0],
            5:[0, 5.0],
            6:[0, -5.0],
            7:[-5.0, 0],
            8:[-5.0, 5.0]}

shank_length = 0.39
thigh_length = 0.41
leg_length_max = thigh_length + shank_length
stand_leg = "left_leg" or 'right_leg'
my_imu_data_var = Vector3()  
# ---------------------------------------------------------------------------------
alg_flag = float(input(' Step Length: \n \
                -1:Move backwards with step_length=-0.35cm\n \
                0:No steps\n \
                1:fix gait 20cm\n \
                2:fix gait 35cm\n \
                3:fix gait 50cm\n \
                4:fix gait 60cm\n '))              
############################################
if __name__ == "__main__":
    rospy.init_node("listener")

    l_foot_joint_pub = rospy.Publisher('/lle/j_foot_l_position_controller/command', Float64, queue_size=10)
    r_foot_joint_pub = rospy.Publisher('/lle/j_foot_r_position_controller/command', Float64, queue_size=10)
    l_knee_joint_pub = rospy.Publisher('/lle/j_knee_l_position_controller/command', Float64, queue_size=10)
    r_knee_joint_pub = rospy.Publisher('/lle/j_knee_r_position_controller/command', Float64, queue_size=10)
    l_hip_joint_pub = rospy.Publisher('/lle/j_hip_l_position_controller/command', Float64, queue_size=10)
    r_hip_joint_pub = rospy.Publisher('/lle/j_hip_r_position_controller/command', Float64, queue_size=10)
    reset_simulation = rospy.ServiceProxy('/gazebo/reset_world', Empty)
    reset_joints = rospy.ServiceProxy('/gazebo/set_model_configuration', SetModelConfiguration)
    unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
    pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        
    publish_rate = 1000
    rate = rospy.Rate(publish_rate)
    
     # global tau_value
    i = 0
    external_force = np.array([0, 0])
    signal.signal(signal.SIGTERM, cb)
    signal.signal(signal.SIGINT, cb)
    trajectory = ld.loadTrajectory("swing_trajectory.txt")
    trajectory_x = trajectory[:, 0]
    trajectory_y = trajectory[:, 1]
    # plt.plot(trajectory_x, trajectory_y, 'r--', lw=2)
    dmp = dmp_discrete.DMPs_discrete(dmps=2, bfs=100)
    dmp.imitate_path(y_des=np.array([trajectory_x, trajectory_y]))
    dmp.goal[0] = -0.2
    gait_time = dmp.goal[0] * 2.3
    if -0.7 < dmp.goal[0] < 0 :
        gait_time = -gait_time
    # tau_value = 1 / gait_time
    # gait_time = 1
    dmp.y0[0] = 0.0
    # step_length = dmp.goal[0]
    dmp.reset_state()
    data_point_num = 3
    # tau_value = (1 / dmp.dt)/data_point_num
    # standard_tau = (1 / dmp.dt)/data_point_num
    tau_value = 1/gait_time

    RL = net.PolicyGradient(
    n_actions=9,
    n_features=5,
    learning_rate=0.02,
    reward_decay=0.99,
    )
    observation = np.array([0, 0, 0, 0, 0])
    i_episode = 0
    batch = 0
    leg = 0

    def imucallback(msg):
        global my_imu_data_var
        # print('llllllllllll', msg.linear_acceleration)
        my_imu_data_var = msg.linear_acceleration #% rospy.get_time()
        return  my_imu_data_var
    def listener():
        # gait_time = 1
        my_imu_data_var
        rospy.init_node('listener')
        rospy.Subscriber('/imu' , Imu , imucallback)
        IMU_pub = rospy.Publisher('/my_imu_acc_data', Vector3, queue_size=10)
        rate = rospy.Rate(1000)
        IMU_pub.publish(my_imu_data_var)
        rate.sleep()

        # ddx0 = my_imu_data_var.x  # if ddx = ddx0
        g = 9.806
        zc = 0.79
        Tc = math.sqrt(zc/g)
        t = 0.5
        tau = t/Tc
        # x0 = ddx0/(zc/g)
        x0= my_imu_data_var.x *2 / (g/zc)
        dx0 = x0 / Tc
        ### states is 1,2,5,6 : x = -Tc*dx
        if x0 < 0 :
            dx0 = -x0 / Tc

        x = x0*math.cosh(tau) + Tc*dx0*math.sinh(tau)
        dx = (x0/Tc)*math.sinh(tau) + dx0*math.cosh(tau)
        ddx  = (g/zc) * x
        # e = 0.5*dx**2-0.5*(g/zc)*x**2
        print('x0',x0 ,'dx0', dx0)
        #----------------------------------------------
        global i, external_force, RL, observation, i_episode , dmp , tau_value, recorder, i, j, \
            step_length, data_point_num, t1, l1, standard_tau, alg_flag, gait_time, batch, leg
    #S: x dx ddx y[0] y[1]
    #A: dmp.goal[0](-0.7 0.7) tau(0.02,10.0) external_force[0] external_force[1]
    #R: down -10 not down 1

        action = RL.choose_action(observation)
        # if i%20 ==0:
            # print('act:', action)
        external_force = np.array(action_set[action])
        y, dy, ddy = dmp.step(tau=tau_value, external_force=external_force)
        # lip.set_swing_foot_pos(y[0], y[1])
        #将观测，动作和回报存储起来
    #-----------------------------------------------
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
            # t = 0.0
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

            # rospy.wait_for_service('/gazebo/pause_physics')
            # try:
            #     pause()
            # except (rospy.ServiceException) as e:
            #     print ("rospause failed!'")
            # set_robot_state()
            # print "called reset()"
    #########################################################
        def update_orbital_energy():
            g = 9.806
            zc = 0.79
            e = (dx**2 - (g / zc) * x**2) / 2
        e = 0.5*dx**2-0.5*(g/zc)*x**2
        print('E:', e)
        def update():
            step_number = 0
            step_number = step_number + 1
            t = 0.5 
            dt = 0.001
            t = t + dt
            # update_motion_state()
            # update_motion_state_recur(f)
            # update_knee_pos()
            update_orbital_energy()
            # update_leg_Length()

            # if stand_leg == 'left_leg':
            #     x = right_foot_x
            #     z = right_foot_z
            # else:
            #     x = left_foot_x
            #     z = left_foot_z
            observation = [x, dx, ddx, x0, zc]
            # print(observation)
            reward = 0.0
            done = 0
            out_range_left   = 0
            out_range_right = 0
            fall = 0  
            out_range = out_range_left or out_range_right
            if step_number >= 5000:
                done = 1
            if fall == 1:
                reward = -10.0
                done = 2
            elif out_range == 1:
                done = 3
                reward = -5.0
            # else:
                # reward = step_number * 0.00001
            info = step_number
            # print('done', done, 'step_number', step_number,'fail',fall,'out_range',out_range)
            return np.array(observation), reward, done, info
    ##########################################################
        observation_, reward, done, info = update()
        # rospy.wait_for_service('/gazebo/unpause_physics')
        # try:
        #     unpause()
        # except (rospy.ServiceException) as e:
        #     print ("/gazebo/pause_physics service call failed")

        if y[1] <= 0.0:
        # if lip.stand_leg == 'left_leg':
        #     lip.switch_support_leg('right_leg')
        #     dmp.y0[0] = lip.left_foot_x
        # else:
        #     lip.switch_support_leg('left_leg')
        #     dmp.y0[0] = lip.right_foot_x
            update_orbital_energy()
            # print('ee:', e)
            reward = 1.0/((e - 0.02)**2+0.005)
            # print('reward=',reward)
            dmp.goal[0] = 0.3# set the goal_x
            gait_time = dmp.goal[0] *2.3
            if -0.7 < dmp.goal[0] < 0 :
                gait_time = -gait_time
            tau_value = 1 / gait_time
            dmp.reset_state()
        i = i + 1

        RL.store_transition(observation, action, reward)
        observation = observation_
        # if done:
        batch = batch + 1
        if batch == 7:
            ep_rs_sum = sum(RL.ep_rs)
            reset(x0, dx0, 0.79, 0.001)
            reset_gazebo()
            ### for learn
            # alpha = np.random.uniform(-10 , -0.1)
            # XhatCom = np.random.uniform(0.1 , 1)
            # ### data sensor
            # Xcom = 0.2
            # XSP = 0.6
            dmp.goal[0] = 0.2 # set the goal_x
            gait_time = dmp.goal[0] *2.3
            if -0.7 < dmp.goal[0] < 0 :
                    gait_time = -gait_time
            tau_value = 1 / gait_time
            dmp.y0[0] = x0 # set the initial x
            # dmp.goal[0] =  0 + alpha * (XhatCom-(Xcom/XSP))
            dmp.reset_state()
            running_reward = ep_rs_sum
            i_episode = i_episode + 1
            batch = 0
            print("episode:", i_episode, "rewards:", (running_reward))
            # print('alpha:', alpha , 'XhatCom:', XhatCom )
            #每个episode学习一次
            vt = RL.learn()
            #智能体探索一步
#S: x dx ddx y[0] y[1] dy[0] dy[1]
#A: dmp.goal[0](-0.7 0.7) tau(0.02,10.0) external_force[0] external_force[1]
#R: down -10 not down 1
 ########################################################
        print('goal.y_track',dmp.goal[0])
        y_track, dy_track, ddy_track = dmp.rollout(tau=tau_value)
        print('tau_value.y_track',tau_value)
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
        # x0 = 0.1
        # dx0 = 0.1
        z0 = 0.79

        x = np.zeros(t.shape)
        dx = np.zeros(t.shape)
        z = np.zeros(t.shape)

        for i in range(ts):
            x[i],dx[i], z[i] = com_tarj(x0, dx0, z0, t[i])
    #####################################
    # y_track, dy_track, ddy_track = dmp.rollout(tau=tau_value)
        x_foot_swing_ref = y_track[:, 0]
        z_foot_swing_ref = y_track[:, 1]

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
        t=0
        if leg == 1 :
            while t <ts :
                l_knee_joint_pub.publish(math.radians(theta_knee_l[t]))
                r_knee_joint_pub.publish(math.radians(theta_knee_r[t]))
                l_hip_joint_pub.publish(math.radians(theta_hip_l[t]))
                r_hip_joint_pub.publish(math.radians(theta_hip_r[t]))
                t = t +1 
                rate.sleep()
                
        # if stand_leg == 'right_leg':
        t=0
        if leg == 2:
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

        zcom = np.random.uniform(0.77 , 0.8)

        listener()
        # print('goal=gait_length:' , dmp.goal[0])
      