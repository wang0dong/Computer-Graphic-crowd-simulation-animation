#!/usr/bin/env python
# coding: utf-8

import cvxopt
import time
from cvxopt import matrix, printing
import numpy
import math
from vedo import *
from random import randrange
from random import randint
# import random
import matplotlib
import matplotlib.pyplot as plt 
import time
# import timeit
# import warnings
# from vtkplotter import *

# export QT_LOGGING_RULES="*.warning=false"

dataurl = "https://vedo.embl.es/examples/data/"

def unique_rand(inicial, limit, total):

        data = []

        i = 0

        while i < total:
            number = randint(inicial, limit)
            if number not in data:
                data.append(number)
                i += 1

        return data

# Define a function to catch warnings
def warn_on_divide_zero(message, category, filename, lineno, file=None, line=None):
    # Check if the warning is a RuntimeWarning and contains "divide by zero"
    if issubclass(category, RuntimeWarning) and "divide by zero" in str(message):
        print("Warning: Division by zero encountered!")

def plot_arrow(x_pos, y_pos, x_direct, y_direct):
    # Creating plot
    fig, ax = plt.subplots(figsize = (12, 7))
    ax.quiver(x_pos, y_pos, x_direct, y_direct)
    ax.set_title('Quiver plot with one arrow')
    named_tuple = time.localtime() # get struct_time
    time_string = time.strftime("%m%d%Y%H%M%S", named_tuple)
    plt.savefig('GradientMap/SingleGradient_%s.png' %time_string)
    plt.close()    

class pedestrian_sim():
    '''
        class definition
    ''' 
    def __init__(self, Experiment):
        '''
            Initialize object
        ''' 
        self.Experiment = Experiment
        if Experiment == 1:
            self.axeslimit = 30
            self.goal = [20, 0, 5]
        else:
            self.axeslimit = 60
            self.goal = [30, -40, 5]
        self.view_angle = [1,0.1,0.5]
        self.ped_rad = 1
        self.step = 1
        self.height = 3
        self.N = 8 # maximum 8
        self.floor1 = [60, 40]
        self.floor2 = [100, 100]
        self.regression_error = 10
        self.regression_iteration = 1000
        self.lazy = 1
        self.crazy_penalty = 9 # 99
        self.obstacle_factor = 1


    def create_world(self):
        height = self.height
        ped_rad = self.ped_rad
        if self.Experiment == 1:
            wall = Box(pos=(5, 11, height), size=(2, 18, height*2), c='red5') + Box(pos=(5, -11, height), size=(2, 18, height*2), c='red5') # walls
            floor = Box(pos=(0, 0, 0), size=(self.floor1[0], self.floor1[1], 2), c='blue5')    # floor
            static_obstacle = Cylinder(pos=(-3, 0, height), r = self.ped_rad*2, height=height*2, axis=(0, 0, 1), c='red5')
            world = wall + floor + static_obstacle
            # obstacle_pos = []
            obstacle_pos = [[(-5, 0, height), ped_rad*4],\
                            [(5, -20, height), ped_rad*2], [(5, -19, height), ped_rad*2], [(5, -18, height), ped_rad*2], [(5, -17, height), ped_rad*2],\
                            [(5, -16, height), ped_rad*2], [(5, -15, height), ped_rad*2], [(5, -14, height), ped_rad*2], [(5, -13, height), ped_rad*2],\
                            [(5, -12, height), ped_rad*2], [(5, -11, height), ped_rad*2], [(5, -10, height), ped_rad*2], [(5, -9, height), ped_rad*2],\
                            [(5, -8, height), ped_rad*2], [(5, -7, height), ped_rad*2], [(5, -6, height), ped_rad*2], [(5, -5, height), ped_rad*2],\
                            [(5, -4, height), ped_rad*2], [(5, -3, height), ped_rad*2],\
                            [(5, 20, height), ped_rad*2], [(5, 19, height), ped_rad*2], [(5, 18, height), ped_rad*2], [(5, 17, height), ped_rad*2],\
                            [(5, 16, height), ped_rad*2], [(5, 15, height), ped_rad*2], [(5, 14, height), ped_rad*2], [(5, 13, height), ped_rad*2],\
                            [(5, 12, height), ped_rad*2], [(5, 11, height), ped_rad*2], [(5, 10, height), ped_rad*2], [(5, 9, height), ped_rad*2],\
                            [(5, 8, height), ped_rad*2], [(5, 7, height), ped_rad*2], [(5, 6, height), ped_rad*2], [(5, 5, height), ped_rad*2],\
                            [(5, 4, height), ped_rad*2], [(5, 3, height), ped_rad*2]
                            ]
        elif self.Experiment == 2:
            wall = Box(pos=(-15, -23, height), size=(68, 2, height*2), c='red5') + \
                    Box(pos=(-29, -10, height), size=(40, 2, height*2), c='red5') + \
                    Box(pos=(9, -10, height), size=(20, 2, height*2), c='red5') + \
                    Box(pos=(0, 20, height), size=(2, 60, height*2), c='red5') + \
                    Box(pos=(-10, 20, height), size=(2, 60, height*2), c='red5') # walls

            floor = Box(pos=(0, 0, 0), size=(self.floor2[0], self.floor2[1], 2), c='blue5')    # floor
            world = wall + floor
            obstacle_pos = []
            for idx in range(-50, 19):
                obstacle_pos.append([(idx, -20, height), ped_rad*2])
            for idx in range(-50, -10):
                obstacle_pos.append([(idx, -10, height), ped_rad*2])
            for idx in range(0, 20):
                obstacle_pos.append([(idx, -10, height), ped_rad*2])
            for idx in range(-10, 50):
                obstacle_pos.append([(-10, idx, height), ped_rad*2])
            for idx in range(-10, 50):
                obstacle_pos.append([(0, idx, height), ped_rad*2])

        return world, obstacle_pos

    def init_pos(self):
        # number of pedestrians
        ped_sep = self.ped_rad * 2
        N = self.N
        num = range(N)
        if self.Experiment == 1:
            xs = unique_rand(int(-25/ped_sep), int(-10/ped_sep), N)
            ys = unique_rand(int(-15/ped_sep), int(15/ped_sep), N)
            pts = [(xs[i]*ped_sep, ys[i]*ped_sep, self.height) for i in num]
        
        elif self.Experiment == 2:
            pts1 = [(randrange(-40, -20, ped_sep), randrange(-20 + ped_sep, -10, ped_sep), self.height) for i in num]
            pts2 = [(randrange(-10 + ped_sep , 0, ped_sep), randrange(10, 30, ped_sep), self.height) for i in num]
            pts = pts1 + pts2
        # print(pts)
        return pts

    def create_ped(self, ped_pos):
        ped_rad = self.ped_rad
        # pedestrian = Spheres(ped_pos, r=4, c="lb", res=8)
        pedestrian = Spheres(ped_pos, r=0, c="lb", res=8)
        # pedestrian's flag
        fss = []    
        for idx in range(len(ped_pos)):
            # man object
            man3d = Mesh(dataurl+'man.vtk')
            man3d.rotate_z(90).scale(2)
            fs = man3d.flagpost(f"{idx}", ped_pos[idx], c='white', alpha=0.5, s=0.8)
            fss.append(fs)
            pedestrian += man3d.c('lb').lighting('glossy').pos(ped_pos[idx])
            # Cylinder object
            # man = Cylinder(pos = ped_pos[idx], r=ped_rad, height=self.height*2, axis=(0, 0, 1), c="lb")
            # fs = man.flagpost(f"{idx}", ped_pos[idx], c='white')
            # fss.append(fs)
            # pedestrian += man

        return pedestrian, fss

    def cost(self, e, obstacles):
        """
        cost function C(x) = |x - g| + sum(f(|x-o_i|)
        where x is the current location of the animated object, g is the goal location, 
        o_i is the location of obstacle , and f is a penalty field for collision avoidance.
        """

        # Register the function to catch RuntimeWarning
        start = time.time()
        # warnings.showwarning = warn_on_divide_zero
        goal = self.goal
        R_e = self.ped_rad
        # |x - g|
        distance2goal = abs(numpy.linalg.norm(np.array(goal) -np.array(e)))
        # distance2goal = math.log(distance2goal)

        # sum(f(|x-o_i|)
        # f(d) = ln(R/d) when 0 < d < (R_o + R_e)*2 or
        # f(d) = 0 when d > (R_o + R_e)*2
        penalty = 0
        for idx in range(len(obstacles)):
            
            obstacle_pos = obstacles[idx][0]
            obstacle_R = obstacles[idx][1]

            distance2obstacle = abs(numpy.linalg.norm(np.array(obstacle_pos) -np.array(e)))

            if distance2obstacle > (obstacle_R + R_e)*1 :
                penalty += 0
            else:
                # penalty += math.log((obstacle_R + R_e)*1/distance2obstacle)
                if distance2obstacle == 0.0:
                    penalty += self.crazy_penalty
                else:
                    penalty += math.log((obstacle_R + R_e)*1/distance2obstacle) * self.obstacle_factor

        end = time.time()
        if end - start > self.lazy:
            print(f"cost function time elapse: {end - start}")

        return (distance2goal + penalty)

    def jacobian_matrix(self, e, obstacles):
        start = time.time()

        goal = self.goal
        R_e = self.ped_rad
        # step = 0.5
        step = self.step
        current_cost = self.cost(e, obstacles)
        # x gradient
        e_delta_x = tuple(x + y for x, y in zip(e, (step, 0 , 0)))
        cost_x_derive = self.cost(e_delta_x, obstacles) - current_cost
        cost_x_derive = cost_x_derive / step
        # y gradient
        e_delta_y = tuple(x + y for x, y in zip(e, (0, step , 0)))
        cost_y_derive = self.cost(e_delta_y, obstacles) - current_cost
        cost_y_derive = cost_y_derive / step

        jacobian = np.array([cost_x_derive, cost_y_derive])

        end = time.time()
        if end - start > self.lazy:
            print(f"jacobian_matrix function time elapse: {end - start}")

        return -1 * jacobian


    def gradient(self, obstacles):
        start = time.time()

        # goal = self.goal
        # R_e = self.ped_rad
        step = self.step


        if self.Experiment == 1:
            feature_x = np.arange(-(self.floor1[0]/2), (self.floor1[0]/2), step)
            feature_y = np.arange(-(self.floor1[1]/2), (self.floor1[1]/2), step)
        else:
            feature_x = np.arange(-(self.floor2[0]/2), (self.floor2[0]/2), step)
            feature_y = np.arange(-(self.floor2[1]/2), (self.floor2[1]/2), step)
    
        X, Y = np.meshgrid(feature_x, feature_y)
        # Define vector field components
        U = np.zeros((len(feature_y), len(feature_x))) # x-component
        V = np.zeros((len(feature_y), len(feature_x))) # y-component
        Z = np.zeros((len(feature_y), len(feature_x))) # cost-component

        for id_y in range(len(feature_y)):
            for id_x in range(len(feature_x)):
                e = np.array([feature_x[id_x], feature_y[id_y], self.height])
                jacobian = self.jacobian_matrix(e, obstacles)

                V[id_y][id_x] = jacobian[1]
                U[id_y][id_x] = jacobian[0]
                Z[id_y][id_x] = self.cost(e, obstacles)


        # Normalize all gradients to focus on the direction not the magnitude
        norm = np.linalg.norm(np.array((U, V)), axis=0)
        U = U / norm
        V = V / norm

        end = time.time()
        if end - start > self.lazy:
            print(f"Gradient function time elapse: {end - start}")

        return [X, Y, U, V, Z]

    def gradient_map(self, gradient, ped_pos):
        # gradient map
        X = gradient[0]
        Y = gradient[1] 
        U = gradient[2] 
        V = gradient[3] 
        Z = gradient[4]

        if self.Experiment == 1:
            fig_width = self.floor1[0] / 2
            fig_height = self.floor1[1] / 2
        else:
            fig_width = self.floor2[0] / 2
            fig_height = self.floor2[1] / 2

        fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))
        ax.set_aspect(1)

        # animate one ped only
        # ped_0 = []
        # for idx in range (len(ped_pos)):
        #     ped_0.append(ped_pos[idx][0])
        # ped_0_pos_x = [coord[0] for coord in ped_0]
        # ped_0_pos_y = [coord[1] for coord in ped_0]
        # ax.plot(ped_0_pos_x, ped_0_pos_y, marker='o', linestyle='-', c='red', alpha=0.5, linewidth=5)

        # animate all ped
        if self.Experiment == 1:
            N = self.N
        else:
            N = self.N * 2
        for idx in range(N):
            ped = []
            for i in range (len(ped_pos)):
                ped.append(ped_pos[i][idx])
            ped_pos_x = [coord[0] for coord in ped]
            ped_pos_y = [coord[1] for coord in ped]
            ax.plot(ped_pos_x, ped_pos_y, marker='o', linestyle='-', c='red', alpha=0.5, linewidth=5)            
            idx += 1

        ax.quiver(X, Y, U, V, units='xy', scale= 2, color='gray')
        # CS = ax.contour(X, Y, Z, 10, cmap='jet', lw=2)
        CS = ax.contour(X, Y, Z, 10, cmap='jet')
        ax.clabel(CS, fontsize=9, inline=True)
        # plt.show()

        named_tuple = time.localtime() # get struct_time
        time_string = time.strftime("%m%d%Y%H%M%S", named_tuple)
        plt.savefig('GradientMap/GradientMap_%s.jpg' %time_string, dpi = 50)
        plt.close()

    def animation_map(self, ped_pos, world):

        pedestrian, fss = self.create_ped(ped_pos)
        ball = Sphere(self.goal, r=self.ped_rad).c("red")
        axes = Axes(xrange=(-self.axeslimit*1.2, self.axeslimit*1.2), yrange=(-self.axeslimit*1.2, self.axeslimit*1.2), zrange=(0, self.axeslimit*1.2))
        plt = Plotter(bg='beige', bg2='lb', axes=10, offscreen=True, interactive=True)
        plt += world
        plt += pedestrian
        plt += ball
        # plt += man3d
        named_tuple = time.localtime() # get struct_time
        time_string = time.strftime("%m%d%Y%H%M%S", named_tuple)
        plt.show(axes, *fss, viewup = self.view_angle)
        screenshot('GradientMap/AnimationMap_%s.png' %time_string)
   
    def move_determine(self, gradient, ped_pos):
        start = time.time()
        X = gradient[0]
        Y = gradient[1]
        U = gradient[2]
        V = gradient[3]

        # critical!
        y_idx = np.abs(X - ped_pos[0]).argmin()
        x_idx = round(np.abs(Y - ped_pos[1]).argmin() / Y.shape[1])

        # x_int = round(ped_pos[0] // self.step + self.floor1[0]/2) - 1
        # x_remain = ped_pos[0] % self.step
        # y_int = round(ped_pos[1] // self.step + self.floor1[1]/2) - 1
        # y_remain = ped_pos[1] % self.step

        x_derive = U[x_idx][y_idx]
        y_derive = V[x_idx][y_idx]

        # plot_arrow(ped_pos[0], ped_pos[1], x_derive, y_derive)

        next_move = [0,0]
        # next_move[0] = (x_derive * (x_remain/self.step) + x_derive_next * (1 - x_remain/self.step))
        next_move[0] = x_derive

        # next_move[1] = (y_derive * (y_remain/self.step) + y_derive_next * (1 - y_remain/self.step))
        next_move[1] = y_derive

        # print(f"ped_x={ped_pos[0]}, ped_y={ped_pos[1]}")
        # print(f"x_derive={x_derive}, y_derive={y_derive}")
        # print(f"x_derive_nxt={x_derive_next}, y_derive_nxt={y_derive_next}")
        # print(f"mov_x={next_move[0]}, mov_y={next_move[1]}")
        # plot_arrow(ped_pos[0], ped_pos[1], next_move[0], next_move[1])

        end = time.time()
        if end - start > self.lazy:
            print(f"move_determine function time elapse: {end - start}")

        return next_move

    # revision 1
    # def calc_dynamic_gradient(self, ped_pos):

    #     dynamic_obstacles = []
    #     for idx in range (len(ped_pos)):
    #         dynamic_obstacles.append([ped_pos[idx], self.height])

    #     dynamic_gradient_lst = []
    #     for idx in range(len(ped_pos)):
    #         temp_dynamic_obstacles = dynamic_obstacles.copy()
    #         temp_dynamic_obstacles.pop(idx) # remove itself from obstacle list
    #         dynamic_gradient = self.gradient(temp_dynamic_obstacles)
    #         dynamic_gradient_lst.append(dynamic_gradient)
    #         temp_dynamic_obstacles = []
        
    #     return dynamic_gradient_lst

    # revision 2
    def calc_dynamic_gradient(self, ped_pos):

        dynamic_obstacles = []
        for idx in range (len(ped_pos)):
            dynamic_obstacles.append([ped_pos[idx], self.height])

        dynamic_gradient_lst = []
        for idx in range(len(ped_pos)):
            temp_dynamic_obstacles = dynamic_obstacles.copy()
            myself = temp_dynamic_obstacles.pop(idx) # remove itself from obstacle list
            for element in temp_dynamic_obstacles: # remove neighbors far away from myself
                if abs(numpy.linalg.norm(np.array(myself[0]) -np.array(element[0]))) > self.ped_rad * 20 :
                    temp_dynamic_obstacles.remove(element)

            dynamic_gradient = self.local_gradient(temp_dynamic_obstacles, myself) # call local gradient function
            dynamic_gradient_lst.append(dynamic_gradient)
            temp_dynamic_obstacles = []
        
        return dynamic_gradient_lst

    def local_gradient(self, obstacles, myself):
        start = time.time()

        step = self.step

        if self.Experiment == 1:
            feature_x = np.arange(-(self.floor1[0]/2), (self.floor1[0]/2), step)
            feature_y = np.arange(-(self.floor1[1]/2), (self.floor1[1]/2), step)
        else:
            feature_x = np.arange(-(self.floor2[0]/2), (self.floor2[0]/2), step)
            feature_y = np.arange(-(self.floor2[1]/2), (self.floor2[1]/2), step)
    
        X, Y = np.meshgrid(feature_x, feature_y)
        # Define vector field components
        U = np.zeros((len(feature_y), len(feature_x))) # x-component
        V = np.zeros((len(feature_y), len(feature_x))) # y-component
        Z = np.zeros((len(feature_y), len(feature_x))) # cost-component

        id_y_low_bound = myself[0][1] - feature_y[0] - self.ped_rad * 10
        if id_y_low_bound < 0:
            id_y_low_bound = 0
        id_y_high_bound = myself[0][1] - feature_y[0] + self.ped_rad * 10
        if id_y_high_bound > len(feature_y):
             id_y_high_bound = len(feature_y) 
        id_x_low_bound = myself[0][0] - feature_x[0] - self.ped_rad * 10
        if id_x_low_bound < 0:
            id_x_low_bound = 0
        id_x_high_bound = myself[0][0] - feature_x[0] + self.ped_rad * 10
        if id_x_high_bound > len(feature_x):
             id_x_high_bound = len(feature_x)             

        for id_y in range(round(id_y_low_bound), round(id_y_high_bound), 1):
            for id_x in range(round(id_x_low_bound), round(id_x_high_bound), 1):
                e = np.array([feature_x[id_x], feature_y[id_y], self.height])
                jacobian = self.jacobian_matrix(e, obstacles)

                V[id_y][id_x] = jacobian[1]
                U[id_y][id_x] = jacobian[0]
                Z[id_y][id_x] = self.cost(e, obstacles)

        # Normalize all gradients to focus on the direction not the magnitude
        norm = np.linalg.norm(np.array((U, V)), axis=0)
        # Compute the median of the non-zero elements
        m = np.median(norm[norm > 0])
        # Assign the median to the zero elements 
        norm[norm == 0] = m
        U = U / norm
        V = V / norm

        end = time.time()
        if end - start > self.lazy:
            print(f"Gradient function time elapse: {end - start}")

        return [X, Y, U, V, Z]

    def inverse_kinematics_gradient_descent(self, ped_pos, obstacles, world):
        start = time.time()

        ped_rad = self.ped_rad
        goal = self.goal
        iteration = 0

        # calculate one time
        static_gradient = self.gradient(obstacles)
        
        ped_pos_history = []
        ped_last_plot = np.array(ped_pos[0]).copy()

        while True:

            # refresh dynamic_obstacle table
            dynamic_obstacles = []
            for idx in range (len(ped_pos)):
                dynamic_obstacles.append([ped_pos[idx], self.height])

            # dynamic_obstacles.pop(0) # ped[0] is himself, dot not take itself as obstacle            
            # dynamic_gradient = self.gradient(dynamic_obstacles)

            # calculate dynamic gradient for ped
            dynamic_gradient_lst = self.calc_dynamic_gradient(ped_pos)

            # move each ped
            for idx in range (len(ped_pos)):
                sum_gradient = (np.array(static_gradient) + np.array(dynamic_gradient_lst[idx]))  
                sum_gradient[0] = sum_gradient[0] /2
                sum_gradient[1] = sum_gradient[1] /2
                next_move = self.move_determine(sum_gradient, ped_pos[idx])
                ped_pos[idx] = (ped_pos[idx][0] + next_move[0], ped_pos[idx][1] + next_move[1], self.height)

            ped_pos_history.append(ped_pos.copy())

            # if abs(numpy.linalg.norm(np.array(ped_pos[0]) - np.array(ped_last_plot))) > 2:
            #     ped_last_plot = np.array(ped_pos[0]).copy()
            #     self.gradient_map(sum_gradient, ped_pos_history)
            #     self.animation_map(ped_pos, world)

            iteration += 1
            self.gradient_map(sum_gradient, ped_pos_history)
            # self.gradient_map(static_gradient, ped_pos_history)
            self.animation_map(ped_pos, world)

            print(f"Distance to goal:{abs(numpy.linalg.norm(np.array(ped_pos[0]) - np.array(goal)))}")

            # gradient descent exit criteria 1
            # all ped reach the goal
            flag = 0
            for idx in range (len(ped_pos)):
                if abs(numpy.linalg.norm(np.array(ped_pos[idx]) - np.array(goal))) < self.regression_error:
                    flag += 1
            if flag == len(ped_pos):
                break

            # if abs(numpy.linalg.norm(np.array(ped_pos[0]) - np.array(goal))) < self.regression_error:
            #     break

            # gradient descent exit criteria 2
            if iteration > self.regression_iteration:
                break


def main():

    experiment = pedestrian_sim(1) # change experiment index
    world, obstacles = experiment.create_world()
    ped_pos = experiment.init_pos()
    experiment.inverse_kinematics_gradient_descent(ped_pos, obstacles, world)

    # pedestrian, fss = experiment.create_ped(ped_pos)
    # ball = Sphere(experiment.goal, r=experiment.ped_rad).c("red")
    # axes = Axes(xrange=(-experiment.axeslimit*1.2, experiment.axeslimit*1.2), yrange=(-experiment.axeslimit*1.2, experiment.axeslimit*1.2), zrange=(0, experiment.axeslimit*1.2))
    # plt = Plotter(bg='beige', bg2='lb', axes=10, offscreen=False, interactive=True)
    # plt += world
    # plt += pedestrian
    # plt += ball
    # # plt += man3d
    # plt.show(axes, *fss, viewup = experiment.view_angle)

if __name__ == '__main__':
    main()
