# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 18:19:59 2024

@author: abombelli
"""

import numpy as np
import pandas as pd
import os
import networkx as nx
import random
import matplotlib as mp
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')

from pyomo.environ import *

############
### Sets ###
############

# Jobs
N = {1:{'P':3,'D':20,'Name':'1, sword','color':"#e88900"},
     2:{'P':2,'D':20,'Name':'2, sword','color':"#e88900"},
     3:{'P':2,'D':20,'Name':'3, sword','color':"#e88900"},
     4:{'P':2,'D':20,'Name':'1, helmet','color':"#e64300"},
     5:{'P':4,'D':20,'Name':'2, helmet','color':"#e64300"},
     6:{'P':1,'D':20,'Name':'3, helmet','color':"#e64300"},
     7:{'P':4,'D':20,'Name':'2, shield','color':"#6b0a1f"},
     8:{'P':3,'D':20,'Name':'3, shield','color':"#6b0a1f"}
     }

# Machines
M = {1:{'Name':'Ironsmith'},
     2:{'Name':'Carpenter'},
     3:{'Name':'Ceramist'}}

# Hierarchy between jobs: (i,j) means
# job i must be completed before j can start
# regardless of which machines they are
# assigned to
N_prec = [(1,2),(2,3),(4,5),(5,6),(7,8)]

# Pre-assigning jobs to machines
N_m = {1:[1,4],
        2:[2,5,7],
        3:[3,6,8]}



model = ConcreteModel()

# Define sets
model.Jobs     = Set(initialize=N.keys())
model.Machines = Set(initialize=M.keys())
model.C_max    = Set(initialize=[0])

# Define parameters
model.P = Param(model.Jobs, initialize={k:v['P'] 
       for k,v in N.items()}, within=Any)
model.D = Param(model.Jobs, initialize={k:v['D'] 
       for k,v in N.items()}, within=Any)

# Define decision variables
model.t     = Var(model.Jobs, within=NonNegativeReals)
model.c     = Var(model.Jobs, within=NonNegativeReals)
model.c_max = Var(model.C_max, within=NonNegativeReals)
model.y     = Var(model.Jobs,model.Jobs, within=Binary)
model.x     = Var(model.Jobs,model.Machines, within=Binary)

# Define objective function
model.obj = Objective(expr=sum(model.c_max[c] for c in model.c_max), sense=minimize)

# Define constraints
model.job_assigned_once = ConstraintList()
for i in model.Jobs:
    model.job_assigned_once.add(expr=sum(model.x[i,m] 
                                         for m in model.Machines)==1)

model.job_assigned_to_machine = ConstraintList()
for m in N_m.keys():
    for n in N_m[m]:
        model.job_assigned_to_machine.add(model.x[n,m]==1)
        
    
model.completion_time = ConstraintList()
for i in model.Jobs:
    model.completion_time.add(model.c[i] >= model.t[i]+model.P[i])
    
model.min_start_time = ConstraintList()
for i in model.Jobs:
    model.min_start_time.add(model.t[i] >= 1)
    
model.max_completion_time = ConstraintList()
for i in model.Jobs:
    model.max_completion_time.add(model.c_max[0] >= model.c[i])

bigM = 200    
model.time_precedence_1 = ConstraintList()
for i in model.Jobs:
    for j in model.Jobs:
        if (i,j) not in N_prec and j>i:
            for m in model.Machines:
                model.time_precedence_1.add(model.t[i]+model.P[i]-model.t[j] 
                                      + bigM*model.y[i,j]+bigM*model.x[i,m]+ 
                                      bigM*model.x[j,m] <= 3*bigM)


model.time_precedence_2 = ConstraintList()
for i in model.Jobs:
    for j in model.Jobs:
            if (i,j) not in N_prec and j>i:
                model.time_precedence_2.add(model.t[j]+model.P[j]-model.t[i]
                                      - bigM*model.y[i,j] <= 0)


model.time_precedence_given = ConstraintList()
for i in model.Jobs:
    for j in model.Jobs:
        if (i,j) in N_prec:
            model.time_precedence_given.add(model.t[i]+model.P[i]<=model.t[j])

model.upper_bound_t = ConstraintList()
for i in model.Jobs:
    model.upper_bound_t.add(model.t[i]<=model.D[i]-model.P[i])
    
model.upper_bound_c = ConstraintList()
for i in model.Jobs:
    model.upper_bound_c.add(model.c[i]<=model.D[i])
    

model.write('PMSP.lp', io_options={'symbolic_solver_labels': True})

# Solve the problem
solver = SolverFactory('gurobi')
solver.solve(model)  

#%%

# Print the results
print('')
print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
print('Overall assignment cost:', model.obj())
print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
print('')

print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
print('Assignment of jobs to machines:')
print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
N_m = {k:[] for k in M.keys()}
for i in model.Jobs:
    for m in model.Machines:
        if model.x[(i,m)].value >= 0.99:
            print(f'Job {i} - Machine {m}')
            N_m[m].append(i)

print('')

print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
print('Start of processing time of jobs:')
print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
T_i = {}
for i in model.Jobs:
    print(f'Job {i}: {model.t[i].value}')
    T_i[i] = model.t[i].value
    
#%%
#########################
### Plotting solution ###
#########################
from matplotlib.patches import Rectangle
axis_font  = {'fontname':'Arial', 'size':'15'}

random.seed(42)
H = 1

plt.close('all')
fig, ax = plt.subplots()
for m in M.keys():
    if len(N_m[m])>0:
        
        for i in N_m[m]:
            ax.add_patch(Rectangle((T_i[i],len(M)*H-m),N[i]['P'],H,
                 edgecolor = 'green',
                 facecolor =  N[i]['color'],
                 fill=True,
                 lw=3))
            plt.text((2*T_i[i]+N[i]['P'])/2,
                     len(M)*H-m+H/2,
                     N[i]['Name'],fontsize=10,color='k')
           
ax.set_xlim(1,model.obj())
ax.set_ylim(0,max(M.keys()))

x_ticks_pos    = [k+0.5 for k in range(1,int(model.obj()))]
x_ticks_labels = [k for k in range(1,int(model.obj()))]


y_ticks_pos    = [len(M)*H-m+H/2 for m in M.keys()]
y_ticks_labels = [m['Name'] for m in M.values()]

ax.set_xticks(x_ticks_pos,labels=x_ticks_labels,rotation=0)

minor_ticks = np.arange(1,model.obj()+1,1)
ax.set_xticks(minor_ticks, minor=True)

ax.set_yticks(y_ticks_pos,labels=y_ticks_labels,rotation=45)
ax.set_xlabel('Day',**axis_font)
ax.set_ylabel('Craftsperson',**axis_font)
ax.grid(which='minor', alpha=0.9)
plt.show()
fig.savefig('sol_PMSP.png', format='png', dpi=1000, bbox_inches='tight',
         transparent=True,pad_inches=0.02) 