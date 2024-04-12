import numpy as np
import matplotlib.pyplot as plt

def map():
    # Start, goal : [x, y, theta]
    start = [0.0, 0.0, 0.0]
    goal = [10.0, 12.0, np.pi/2]
    # Searching space : [min_x, max_x, min_y, max_y]
    space = [-2.0, 15.0, -2.0, 15.0]  
    # Obstacle : (x, y, radius)
    obstacle_list = [[3, 10, 1],
                     [3, 11, 1],
                     [3, 12, 1],
                     [3, 13, 1],
                     [3, 14, 1],
                     [7, 10, 1],
                     [7, 11, 1],
                     [7, 12, 1],
                     [7, 13, 1],
                     [7, 14, 1]]
    
    return start, goal, obstacle_list, space

if __name__ == "__main__":
    theta_plot = np.linspace(0,1,101) * np.pi * 2
    start, goal, obstacle_list, space = map()
    plt.figure(figsize=(8,8))
    plt.plot(start[0], start[1], 'bs',  markersize=7)
    plt.text(start[0], start[1]+0.5, 'start', fontsize=12)
    plt.plot(goal[0], goal[1], 'rs',  markersize=7)
    plt.text(goal[0], goal[1]+0.5, 'goal', fontsize=12)
    for i in range(len(obstacle_list)):
        x_obstacle = obstacle_list[i][0] + obstacle_list[i][2] * np.cos(theta_plot)
        y_obstacle = obstacle_list[i][1] + obstacle_list[i][2] * np.sin(theta_plot)
        plt.plot(x_obstacle, y_obstacle,'k-')
    plt.axis(space)
    plt.grid(True)
    plt.show()