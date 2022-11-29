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
from os import listdir
from os.path import isfile, join
import PIL

cwd = os.getcwd()
plt.close('all')

# Set random seed
random.seed(42)

# Loading input file
df = pd.read_excel(os.path.join(cwd,'Knight_network.xlsx'), keep_default_na=False)

# Define unique set of nodes
Nodes = sorted(list(set(df.Origin) | set(df.Destination)))
N     = {Nodes[idx]:idx for idx,n in enumerate(Nodes)}

# Define edges (we have one distinct edge per row of the input dataframe)
E = {}
for index, row in df.iterrows():
    E[row.Origin,row.Destination] = [row.Origin,row.Destination,row.Name,row.Cost]
    
# Define sets of ingoing/outgoing edges per node

N_in  = {node:[E[key][0] for key in E.keys() if E[key][1]==node] for node in N.keys()}
N_out = {node:[E[key][1] for key in E.keys() if E[key][0]==node] for node in N.keys()}

# Define source and sink nodes
source = 'A'
sink   = 'M'

    
#############################
### SP OPTIMIZATION MODEL ###
#############################

print('Setting up model')

start_time = time.time()  

# Setup model
model = Model()

print('Creating decision variables')
x = {}

for key in E.keys():
    x[E[key][0],E[key][1]]=model.addVar(lb=0, ub=1, vtype=GRB.BINARY,name="x[%s,%s]"%(E[key][0],E[key][1]))

model.update()
            
print('Creating constraints')

for n in N.keys():
    lhs = LinExpr()
    for n_out in N_out[n]:
        lhs += x[n,n_out]
    for n_in in N_in[n]:
        lhs -= x[n_in,n]
    if n == source:
        rhs = 1
    elif n == sink:
        rhs = -1
    else:
        rhs = 0
    model.addConstr(lhs=lhs, sense=GRB.EQUAL, rhs=rhs,
    name='flow_cons_[%s]'%(n)) 

print('Creating objective')
    
obj = LinExpr()

for key in E.keys():
    obj += E[key][3]*x[E[key][0],E[key][1]]

model.setObjective(obj,GRB.MINIMIZE)

model.update()


model.write('SP.lp') 

print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
print('Working on a model with %i decision variables'%(len(model.getVars())))
print('Working on a model with %i constraints'%(len(model.getConstrs())))
print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%') 

# Solve
model.setParam('MIPGap',0.001)
model.setParam('TimeLimit',2*3600)
model.params.LogFile='2D_BPP.log'

model.optimize()
endTime   = time.time()  

solution = []

# Retrieve variable names and values
for v in model.getVars():
    solution.append([v.varName,v.x])

active_variables       = []
active_variables_names = []
for i in range(0,len(solution)):
    # Adding binary variables that are unitary
    if solution[i][1] >= 0.99:
        active_variables.append([solution[i][0],solution[i][1]]) 
        active_variables_names.append(solution[i][0])
#%%

# Reconvert to edges
E_solution = []
for act_var in active_variables_names:
    idx_openBr   = [pos for pos, char in enumerate(act_var) if char == '['][0]
    idx_closedBr = [pos for pos, char in enumerate(act_var) if char == ']'][0]
    idx_comma    = [pos for pos, char in enumerate(act_var) if char == ','][0]
    first_node   = act_var[idx_openBr+1:idx_comma]
    second_node  = act_var[idx_comma+1:idx_closedBr]
    E_solution.append((first_node,second_node))
    
# Construct path
Path = []
for e in E_solution:
    Path.append(E[e][2])
    
# Print solution
print('')
print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
print('')
print('Our knight will have to face, in sequence,')
for edge in Path:
    print(edge)
print('on his way to the castle')   

#%%




mypath = os.path.join(cwd,'Figures')

file_names        = [f for f in listdir(mypath) if isfile(join(mypath, f))]
file_names_no_ext = [f.split('.')[0] for f in listdir(mypath) if isfile(join(mypath, f))]

icons = {}
for key in E.keys():
    if E[key][2] in file_names_no_ext:
        icons[E[key][2]] = os.path.join(cwd,'Figures',file_names[file_names_no_ext.index(E[key][2])])
    else:
        icons[E[key][2]] = os.path.join(cwd,'Figures','dummy.jpg')

# Load images
images = {k: PIL.Image.open(fname) for k, fname in icons.items()}

# Images for source and sink nodes
icons_source_sink = {'knight':os.path.join(cwd,'Figures','knight.png'),
         'castle':os.path.join(cwd,'Figures','castle.png'),
    }
images_source_sink = {k: PIL.Image.open(fname) for k, fname in icons_source_sink.items()}

TUD_cyan   = [0/255.0, 166/255.0, 214/255.0]
TUD_orange = [224/255.0, 60/255.0, 49/255.0]

# Plot solution

coord_N = {'A':(0,0),
           'B':(2,8),
           'C':(5,0),
           'D':(9,-8),
           'E':(6,12),
           'F':(6,7),
           'G':(9,15),
           'H':(15,0),
           'I':(13,-3),
           'J':(13,-9),
           'K':(18,6),
           'L':(18,-6),
           'M':(21,0)}

import networkx as nx
G       = nx.DiGraph()
G.coord = {}

for key in N.keys():
    if key == source:
        G.add_node(key,idx=N[key],coord=coord_N[key],image=images_source_sink['knight'])
        G.coord[key] = coord_N[key]
    elif key == sink:
        G.add_node(key,idx=N[key],coord=coord_N[key],image=images_source_sink['castle'])
        G.coord[key] = coord_N[key]
    else:
        G.add_node(key,idx=N[key],coord=coord_N[key])
        G.coord[key] = coord_N[key]
        


#%%     
    
for key in E.keys():
    G.add_edge(E[key][0],E[key][1],name=E[key][2],cost=E[key][3],image=images[E[key][2]])
edge_color = []
edge_width = []
for key in E.keys():
    if key in E_solution:
        edge_color.append('r')
        edge_width.append(3)
    else:
        edge_color.append('gray')
        edge_width.append(.5)

fig, ax = plt.subplots(figsize=(15,12))
nx.draw_networkx(G, ax=ax,
                 font_size=12,
                 alpha=.6,
                 width=.075,
                 pos=G.coord,
                 node_color=[TUD_cyan]*len(N))
nx.draw_networkx_edges(G, G.coord, edge_color=edge_color, width=edge_width, alpha=0.8, arrows=True, arrowstyle='-|>',arrowsize=5) 
edge_labels = dict([(key,E[key][3]) for key in E.keys()]) 
nx.draw_networkx_edge_labels(G, G.coord, edge_labels=edge_labels, label_pos=0.7,
                             font_color=TUD_orange, font_size=10, font_weight='bold')

# Transform from data coordinates (scaled between xlim and ylim) to display coordinates
tr_figure = ax.transData.transform
# Transform from display to figure coordinates
tr_axes = fig.transFigure.inverted().transform

# Select the size of the image (relative to the X axis)
icon_size = (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.005
icon_center = icon_size / 2.0

# Add the respective image to each node
for n in N.keys():
    if n == source:
        xf, yf = tr_figure(coord_N[n])
        xa, ya = tr_axes((xf, yf))
        # get overlapped axes and plot icon
        a = plt.axes([xa - icon_center, ya - icon_center, icon_size, icon_size])
        a.imshow(G.nodes[n]['image'])
    elif n == sink:
        xf, yf = tr_figure(coord_N[n])
        xa, ya = tr_axes((xf, yf))
        # get overlapped axes and plot icon
        a = plt.axes([xa - icon_center, ya - icon_center, icon_size, icon_size])
        a.imshow(G.nodes[n]['image'])
    a.axis("off")
plt.show()

# Select the size of the image (relative to the X axis)
icon_size_edge = (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.002
icon_center_edge = icon_size_edge / 2.0



for e in E.keys():
    
    if 'image' in G.edges[e]:
    
        x_1 = coord_N[e[0]][0]
        x_2 = coord_N[e[1]][0]
        y_1 = coord_N[e[0]][1]
        y_2 = coord_N[e[1]][1]
        xc  = (x_1+x_2)/2 
        yc  = (y_1+y_2)/2
        xf, yf = tr_figure((xc,yc))
        xa, ya = tr_axes((xf, yf))
        a = plt.axes([xa - icon_center_edge, ya - icon_center_edge, icon_size_edge, icon_size_edge])
        a.imshow(G.edges[e]['image'])
 
        a.axis("off")
plt.show()

plt.savefig('SP.png', format='png', dpi=400, bbox_inches='tight',
             transparent=True,pad_inches=0.02)



