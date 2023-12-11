#!/usr/bin/env python3
import numpy as np
import rospy

yy = "swing_trajectory.txt"
def loadTrajectory(file_name):
    file = open(file_name)
    line_len = 0
    for line in file:
        line_len += 1
    trajectory = np.zeros((line_len, 2))
    file.close()

    file = open(yy)
    index = 0
    for line in file:
        item = str.split(line)
        trajectory[index, 0] = (item[0])
        trajectory[index, 1] = (item[1])
        index += 1
    file.close()
    return trajectory


if __name__ == "__main__":
    rospy.init_node("twist_to_motors")
    import matplotlib.pyplot as plt

    trajectory = loadTrajectory("swing_trajectory.txt")
    # print (trajectory)
    rospy.loginfo(trajectory)
    # plt.figure()
    plt.plot(trajectory[:, 0], trajectory[:, 1])
    # plt.show()
    # plt.show(block = True)
    # rospy.spin()

    
