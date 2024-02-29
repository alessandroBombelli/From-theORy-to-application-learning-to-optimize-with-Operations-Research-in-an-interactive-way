# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 13:29:20 2022

@author: abombelli
"""
import numpy as np
import os
import pandas as pd
import time
from gurobipy import Model,GRB,LinExpr,quicksum
import random
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from os import listdir
from os.path import isfile, join
import PIL
import imageio

cwd = os.getcwd()
plt.close('all')

# Set random seed
random.seed(2)

# Define grid where robots move

# Horizontal nodes
N_h = 10
# Vertical nodes
N_v = 5
# Horizontal extension
L_h = 50
# Vertical extension
L_v = 30
# Horizontal separation between nodes
Sep_h = L_h/(N_h-1)
# Vertical separation between nodes
Sep_v = L_v/(N_v-1)

# Define set of nodes. We start our enumeration with 0, which is the top-left
# node, and we progressively continue moving right and then bottom
N_nodes = N_h*N_v 
N       = {}

for i in range (0,N_nodes):
    row    = int(np.floor(i/N_h))
    column = int(np.mod(i-(row+1)*N_h,N_h))
    # Define position of node in the grid
    if row == 0:
        if column == 0:
            pos = 'ULC'
        elif column == N_h-1:
            pos = 'URC'
        else:
            pos = 'U'
    elif row == N_v-1:
        if column == 0:
            pos = 'BLC'
        elif column == N_h-1:
            pos = 'BRC'
        else:
            pos = 'B'
    else:
        if column == 0:
            pos = 'L'
        elif column == N_h-1:
            pos = 'R'
        else:
            pos = 'I'
        
            

    N[i]   = [i,row,column,column*Sep_h,L_v-row*Sep_v,pos]

plot_grid = False

if plot_grid is True:    
    # Plot grid
    fig, ax = plt.subplots()
    for n in N.keys():
        plt.plot(N[n][3],N[n][4],marker='o',markersize=5,color='r')
        plt.text(N[n][3],N[n][4],N[n][0],fontsize=10)
    ax.grid(True)
    plt.show()

######################################    
### Define possible moves per node ###
######################################
N_moves = {}
for n in N.keys():
    # Robot can move to the right or below 
    if N[n][5] == 'ULC':
        N_moves[n] = sorted([n,n+1,n+N_h])
    # Robot can move to the left or below
    elif N[n][5] == 'URC':
        N_moves[n] = sorted([n,n-1,n+N_h])
    # Robot can move to the right or up
    elif N[n][5] == 'BLC':
        N_moves[n] = sorted([n,n+1,n-N_h])        
    # Robot can move to the left or up 
    elif N[n][5] == 'BRC':
        N_moves[n] = sorted([n,n-1,n-N_h])
    # Robot can move to the left or right or down 
    elif N[n][5] == 'U':
        N_moves[n] = sorted([n,n-1,n+1,n+N_h])
    # Robot can move to the left or right or up 
    elif N[n][5] == 'B':
        N_moves[n] = sorted([n,n-1,n+1,n-N_h])   
    # Robot can move up or down or to the right 
    elif N[n][5] == 'L':
        N_moves[n] = sorted([n,n-N_h,n+1,n+N_h])    
    # Robot can move up or down or to the left  
    elif N[n][5] == 'R':
        N_moves[n] = sorted([n,n-N_h,n-1,n+N_h]) 
    # Robot can move up or down or to the left or to the right
    else:
        N_moves[n] = sorted([n,n-N_h,n-1,n+1,n+N_h])
        
##########################
### Find reversed arcs ###
##########################
RA = []
for n1 in N.keys():
    for n2 in N_moves[n1]:
        if n2!=n1 and n1 in N_moves[n2]:
            if (n1,n2) not in RA and (n2,n1) not in RA:
                RA.append((n1,n2))           


#####################################
### Inputs for optimization model ###
#####################################

# Time vector
T_max = 20
T     = list(np.arange(0,T_max))

# Robots
N_r = 5
R   = list(np.arange(0,N_r))
# For each robot, randomly select a distinct initial and final position
R_pos = {}
candidate_nodes = list(N.keys())
for r in R:
    origin = random.choice(candidate_nodes)
    candidate_nodes.remove(origin)
    destination = random.choice(candidate_nodes)
    candidate_nodes.remove(destination)
    R_pos[r] = [origin,destination]
    
########################################
### PATH PLANNING OPTIMIZATION MODEL ###
########################################

additional_constraints = True

print('Setting up model')

start_time = time.time()  

# Setup model
model = Model()

print('Creating decision variables')
z = {}
for r in R:
    for n in N.keys():
        for t in T:
            z[r,n,t]=model.addVar(lb=0, ub=1, vtype=GRB.BINARY,name="z[%s,%s,%s]"%(r,n,t))

w = {}
for r in R:
    # Moves are defined for all time-steps but the last one
    for t in T[:-1]:
        for o in N.keys():
            for d in N_moves[o]:
                w[r,t,o,d]=model.addVar(lb=0, ub=1, vtype=GRB.BINARY,name="w[%s,%s,%s,%s]"%(r,t,o,d))

print('Creating constraints')

# Each robot needs to be exactly in one node at each time-step
for r in R:
    for t in T:
        lhs = LinExpr()
        for n in N.keys():
            lhs += z[r,n,t]
        model.addConstr(lhs=lhs, sense=GRB.EQUAL, rhs=1,
                        name='position_n_t[%s,%s]'%(r,t))

# Each node can be occupied by at most one robot at the same time
for n in N.keys():
    for t in T:
        lhs = LinExpr()
        for r in R:
            lhs += z[r,n,t]
        model.addConstr(lhs=lhs, sense=GRB.LESS_EQUAL, rhs=1,
                        name='max_occup_n_t[%s,%s]'%(n,t))

# Fix initial and final position of all robots
for r in R:
    lhs = LinExpr()
    lhs += z[r,R_pos[r][0],T[0]]
    model.addConstr(lhs=lhs, sense=GRB.EQUAL, rhs=1,
                        name='init_pos_r[%s]'%(r))
    lhs = LinExpr()
    lhs += z[r,R_pos[r][1],T[-1]]
    model.addConstr(lhs=lhs, sense=GRB.EQUAL, rhs=1,
                        name='final_pos_r[%s]'%(r))
    
# Each robot needs to decide a move from its current position
for r in R:
    for n in N.keys():
        # Recall that moves are defined for all time-steps but the last one 
        for t in T[:-1]:
            lhs = LinExpr()
            lhs += z[r,n,t]
            for move in N_moves[n]:
                lhs -= w[r,t,n,move]
            model.addConstr(lhs=lhs, sense=GRB.EQUAL, rhs=0,
                        name='force_move_r_n_t[%s,%s,%s]'%(r,n,t))

# Each arc can be used at most by one robot at any given time
for n1 in N.keys():
    for n2 in N_moves[n1]:
        for t in T[:-1]:
            lhs = LinExpr()
            for r in R:
                lhs += w[r,t,n1,n2]
            model.addConstr(lhs=lhs, sense=GRB.LESS_EQUAL, rhs=1,
                        name='arc_n1_n2_t[%s,%s,%s]'%(n1,n2,t))
                
# Update position of robot
for r in R:
    for n in N.keys():
        # Recall that moves are defined for all time-steps but the last one 
        for t in T[:-1]:
            for move in N_moves[n]:
                lhs = LinExpr()
                lhs += z[r,move,t+1]-z[r,n,t]-w[r,t,n,move]+1
                model.addConstr(lhs=lhs, sense=GRB.GREATER_EQUAL, rhs=0,
                        name='move_r_t_n1_n2_1[%s,%s,%s,%s]'%(r,t,n,move))
                lhs = LinExpr()
                lhs += z[r,move,t+1]-z[r,n,t]+w[r,t,n,move]-1
                model.addConstr(lhs=lhs, sense=GRB.LESS_EQUAL, rhs=0,
                        name='move_r_t_n1_n2_2[%s,%s,%s,%s]'%(r,t,n,move))
                
# Each arc can be used at most in one direction to avoid conflicts
# between robots
for arc in RA:
    for t in T[:-1]:
        lhs = LinExpr()
        for r in R:
            lhs += w[r,t,arc[0],arc[1]]
            lhs += w[r,t,arc[1],arc[0]]
        model.addConstr(lhs=lhs, sense=GRB.LESS_EQUAL, rhs=1,
                    name='arc_rev_n1_n2_t[%s,%s,%s]'%(arc[0],arc[1],t))
                
if additional_constraints is True:
    # Robots cannot occupy at any time neither the origin nor the destination
    # node of other robots (this could be quite restrictive and cause
    # infeasibility)
    for r1 in R:
        origin = R_pos[r1][0]
        dest   = R_pos[r1][1] 
        for r2 in R:
            if r2 != r1:
                lhs = LinExpr()
                for t in T:
                    lhs += z[r2,origin,t]
                model.addConstr(lhs=lhs, sense=GRB.EQUAL, rhs=0,
                        name='forbid_r_n[%s,%s]'%(r2,origin))
                lhs = LinExpr()
                for t in T:
                    lhs += z[r2,dest,t]
                model.addConstr(lhs=lhs, sense=GRB.EQUAL, rhs=0,
                        name='forbid_r_n[%s,%s]'%(r2,dest))
    

    
    
                
                
print('Creating objective')
    
obj = LinExpr()
for r in R:
    for t in T[:-1]:
        for n in N.keys():
            for move in N_moves[n]:
                if n != move:
                    row_1 = N[n][1]
                    row_2 = N[move][1]
                    col_1 = N[n][2]
                    col_2 = N[move][2]
                    
                    if row_1 != row_2:
                        obj += Sep_v*w[r,t,n,move]
                    else:
                        obj += Sep_h*w[r,t,n,move]
                        

model.setObjective(obj,GRB.MINIMIZE)

model.update()

print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
print('Working on a model with %i decision variables'%(len(model.getVars())))
print('Working on a model with %i constraints'%(len(model.getConstrs())))
print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%') 

# Solve
model.setParam('MIPGap',0.001)
# Setting computational time to 30 minutes
model.setParam('TimeLimit',120)
model.write('path_planning.lp') 
model.params.LogFile='path_planning.log'

model.optimize()
endTime   = time.time()  

solution = []

# Retrieve variable names and values
for v in model.getVars():
    solution.append([v.varName,v.x])

# Retrieving active decision variables
active_variables       = []
active_variables_names = []
for i in range(0,len(solution)):
    # Adding binary variables that are unitary
    if solution[i][1] >= 0.99:
        active_variables.append([solution[i][0],solution[i][1]]) 
        active_variables_names.append(solution[i][0])
        
R_act_var = {r:[] for r in R}
for act_var in active_variables_names:
    if act_var[0] == 'z': 
        idx_openBr  = [pos for pos, char in enumerate(act_var) if char == '['][0]
        idx_comma   = [pos for pos, char in enumerate(act_var) if char == ','][0]
        robot_id    = int(act_var[idx_openBr+1:idx_comma])
        R_act_var[robot_id].append(act_var)
        
R_time_ordered_act_var = {r:[] for r in R}
for r in R:
    act_var_r  = R_act_var[r]
    timestamps = []
    for var in act_var_r:
        idx_sec_comma = [pos for pos, char in enumerate(var) if char == ','][1]
        idx_closedBr  = [pos for pos, char in enumerate(var) if char == ']'][0]
        timestamps.append(int(var[idx_sec_comma+1:idx_closedBr]))
    R_time_ordered_act_var[r] = [R_act_var[r][idx] for idx in list(np.argsort(timestamps))] 
    
R_node_time = {r:[] for r in R}
for r in R:
    act_var_r_srt  = R_time_ordered_act_var[r]
    for var in act_var_r_srt:
        idx_first_comma = [pos for pos, char in enumerate(var) if char == ','][0]
        idx_sec_comma   = [pos for pos, char in enumerate(var) if char == ','][1]
        idx_closedBr  = [pos for pos, char in enumerate(var) if char == ']'][0]
        R_node_time[r].append((int(var[idx_first_comma+1:idx_sec_comma]),
                               int(var[idx_sec_comma+1:idx_closedBr])))

#%% 

################################
### Plot final path planning ###
################################

plt.close('all') 

def getImage(path, zoom=.1):
    return OffsetImage(plt.imread(path), zoom=zoom)

# Settings for font
axis_font  = {'fontname':'Arial', 'size':'15'}

# Generate a random color per robot
R_colors = {r:[random.randint(0,255)/255.,random.randint(0,255)/255.,random.randint(0,255)/255.] for r in R}

# Load robot Figure
im = mpimg.imread(os.path.join(cwd,'Figures','robot.png'))
      
# Plot path planning
fig, ax = plt.subplots()


filenames = []
for t in T:
    
    # Plot grid
    for n in N.keys():
        plt.plot(N[n][3],N[n][4],marker='o',markersize=4,color='g')
    for i in range(0,N_h-1):
        for j in range(0,N_v):
            plt.plot([i*Sep_h,(i+1)*Sep_h],[L_v-j*Sep_v,L_v-j*Sep_v],
                     linewidth=1,linestyle='-',color='k')
    for i in range(0,N_v-1):
        for j in range(0,N_h):
            plt.plot([j*Sep_h,j*Sep_h],[i*Sep_v,(i+1)*Sep_v],
                     linewidth=1,linestyle='-',color='k') 
            
    # Plot initial and final position of robots
    for r in R:
        plt.plot(N[R_pos[r][0]][3],N[R_pos[r][0]][4],marker='o',markersize=20,
                 alpha=0.6,color=R_colors[r],markeredgecolor='g',
                 markeredgewidth=2)
        plt.plot(N[R_pos[r][1]][3],N[R_pos[r][1]][4],marker='o',markersize=20,
                 alpha=0.6,color=R_colors[r],markeredgecolor='r',
                 markeredgewidth=2)

    
    for r in R:
        current_node = R_node_time[r][t][0]
        ab = AnnotationBbox(getImage(os.path.join(cwd,'Figures','robot.png')),
                            (N[current_node][3], N[current_node][4]), frameon=False, zorder=10)
        ax.add_artist(ab)
    
    plt.text(0*L_h,1.1*L_v,'Time-step: %s'%(t),fontsize=12,
            bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=0.1'))
    #ax.set_xlabel('X direction',**axis_font)
    #ax.set_ylabel('Y direction',**axis_font)
    plt.draw()
    plt.pause(1)
    
    # create file name and append it to a list
    filename = f'{t}.png'
    filenames.append(filename)
    plt.savefig(os.path.join(cwd,'Figures',filename))
        
    
    if t != T[-1]:
        plt.cla()

# build gif
with imageio.get_writer(os.path.join(cwd,'Figures','mygif.gif'), mode='I',fps=1) as writer:
    for filename in filenames:
        image = imageio.imread(os.path.join(cwd,'Figures',filename))
        writer.append_data(image)
        
# Remove files
for filename in set(filenames):
    os.remove(os.path.join(cwd,'Figures',filename))
    

 






