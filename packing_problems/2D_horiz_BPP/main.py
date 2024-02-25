# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 15:00:54 2024

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

# Bin types
T = {1:{'L':10,'W':7,'C':10,'#':2},
     2:{'L':8,'W':5,'C':8,'#':2}}

# Bins
cont = 1
B    = {}
for t,v in T.items():
    for j in range(0,v['#']):
        B[cont] = {'type':t,'L':v['L'],'W':v['W'],'C':v['C']}
        cont += 1
    
B_t = {t:[k for k,v in B.items() if v['type']==t] for t in T.keys()}

# Items
I = {1:{'L':4,'W':2,'bin_types':[1,2],'rotation':'Y','incomp':[12,15]},
    2:{'L':2,'W':3,'bin_types':[1,2],'rotation':'Y','incomp':[12,16]}, 
    3:{'L':2,'W':5,'bin_types':[1],'rotation':'N','incomp':[]}, 
    4:{'L':5,'W':1,'bin_types':[1],'rotation':'Y','incomp':[14]},
    5:{'L':3,'W':3,'bin_types':[1,2],'rotation':'Y','incomp':[]},
    6:{'L':3,'W':2,'bin_types':[1,2],'rotation':'Y','incomp':[]},
    7:{'L':2,'W':3,'bin_types':[1,2],'rotation':'Y','incomp':[13]},
    8:{'L':5,'W':2,'bin_types':[1,2],'rotation':'Y','incomp':[14]},
    9:{'L':3,'W':1,'bin_types':[1],'rotation':'N','incomp':[]},
    10:{'L':2,'W':1,'bin_types':[1,2],'rotation':'Y','incomp':[]},
    11:{'L':5,'W':1,'bin_types':[1,2],'rotation':'Y','incomp':[15]},
    12:{'L':4,'W':3,'bin_types':[2],'rotation':'N','incomp':[2]},
    13:{'L':4,'W':2,'bin_types':[1,2],'rotation':'Y','incomp':[3]},
    14:{'L':2,'W':1,'bin_types':[1,2],'rotation':'Y','incomp':[4]},
    15:{'L':2,'W':3,'bin_types':[1,2],'rotation':'Y','incomp':[5]},
    16:{'L':2,'W':6,'bin_types':[1,2],'rotation':'Y','incomp':[2,3]}     
     }

B_i = {i:[[b for b in B_t[t]] for t in I[i]['bin_types']] for i in I.keys()}
B_i = {i:[x for xs in v for x in xs] for i,v in B_i.items()} # Flattening list of lists

I_inc = [(i1,i2) for i1,v in I.items() for i2 in v['incomp']]

model = ConcreteModel()

# Define sets
model.Bins     = Set(initialize=B.keys())
model.Items    = Set(initialize=I.keys())
model.XY       = Set(initialize=[1,2])

# Define parameters
model.L_b = Param(model.Bins, initialize={k:v['L'] for k,v in B.items()}, within=Any)
model.W_b = Param(model.Bins, initialize={k:v['W'] for k,v in B.items()}, within=Any)
model.L_i = Param(model.Items, initialize={k:v['L'] for k,v in I.items()}, within=Any)
model.W_i = Param(model.Items, initialize={k:v['W'] for k,v in I.items()}, within=Any)
model.C_b = Param(model.Bins, initialize={k:v['C'] for k,v in B.items()}, within=Any)

L_max = max([v['L'] for k,v in B.items()])
W_max = max([v['W'] for k,v in B.items()])

eps   = 0.1

# Define decision variables
model.x = Var(model.Items, within=NonNegativeReals)
model.y = Var(model.Items, within=NonNegativeReals)
model.r = Var(model.Items,model.XY,model.XY, within=Binary)
model.p = Var(model.Items,model.Bins, within=Binary)
model.l = Var(model.Items,model.Items, within=Binary)
model.b = Var(model.Items,model.Items, within=Binary)
model.z = Var(model.Bins, within=Binary)

# Define objective function
model.obj = Objective(expr=sum(model.C_b[b]*model.z[b] for b in model.Bins), sense=minimize)

# Define constraints

model.no_overlap = ConstraintList()
for i in model.Items:
    for j in model.Items:
        if j != i:
            for b in list(set(B_i[i]) & set(B_i[j])):
                model.no_overlap.add(model.l[i,j]+model.l[j,i]+
                                 model.b[i,j]+model.b[j,i] >= 
                                 model.p[i,b]+model.p[j,b]-1)
                
model.relative_x = ConstraintList()
for i in model.Items:
    for j in model.Items:
        if j != i:
            for b in list(set(B_i[i]) & set(B_i[j])):
                model.relative_x.add(model.x[j]>=model.x[i]+model.L_i[i]*model.r[i,1,1]
                                     +model.W_i[i]*model.r[i,1,2]-L_max*(1-model.l[i,j]))
                model.relative_x.add(model.x[j]+eps<=model.x[i]+model.L_i[i]*model.r[i,1,1]
                                     +model.W_i[i]*model.r[i,1,2]+L_max*model.l[i,j])
                
model.relative_y = ConstraintList()
for i in model.Items:
    for j in model.Items:
        if j != i:
            for b in list(set(B_i[i]) & set(B_i[j])):
                model.relative_y.add(model.y[j]>=model.y[i]+model.L_i[i]*model.r[i,2,1]
                                     +model.W_i[i]*model.r[i,2,2]-W_max*(1-model.b[i,j]))
                model.relative_y.add(model.y[j]+eps<=model.y[i]+model.L_i[i]*model.r[i,2,1]
                                     +model.W_i[i]*model.r[i,2,2]+W_max*model.b[i,j])

model.max_x = ConstraintList()
for i in model.Items:
    model.max_x.add(model.x[i]+model.L_i[i]*model.r[i,1,1]
                    +model.W_i[i]*model.r[i,1,2]<=
                    sum(model.L_b[b]*model.p[i,b] for b in B_i[i]))

model.max_y = ConstraintList()
for i in model.Items:
    model.max_y.add(model.y[i]+model.L_i[i]*model.r[i,2,1]
                    +model.W_i[i]*model.r[i,2,2]<=
                    sum(model.W_b[b]*model.p[i,b] for b in B_i[i]))

model.r1 = ConstraintList()
for i in model.Items:
    for d1 in model.XY:
        model.r1.add(sum(model.r[i,d1,d2] for d2 in model.XY)==1)
        
model.r2 = ConstraintList()
for i in model.Items:
    for d1 in model.XY:
        model.r2.add(sum(model.r[i,d2,d1] for d2 in model.XY)==1)
        
model.item_assignment = ConstraintList()
for i in model.Items:
    model.item_assignment.add(sum(model.p[i,b] for b in B_i[i])==1)
    
model.bin_activation = ConstraintList()
for i in model.Items:
    for b in B_i[i]:
        model.bin_activation.add(model.p[i,b]<=model.z[b])
        
model.symmetry_breaking = ConstraintList()
for t,v in B_t.items():
    for b in range(0,len(v)-1):
        model.symmetry_breaking.add(model.z[v[b+1]]<=model.z[v[b]])
        
model.item_incomp = ConstraintList()
for pair in I_inc:
    for b in list(set(B_i[pair[0]]) & set(B_i[pair[1]])):
        model.item_incomp.add(model.p[pair[0],b]+model.p[pair[1],b]<=1)

# Solve the problem
solver = SolverFactory('gurobi')
solver.solve(model)  

#%%

I_b = {b:[] for b in B.keys()}

# Print the results
print('Overall cost of used bins:', model.obj())
for i in model.Items:
    for b in B_i[i]:
        print(f'Item {i} - Bin {b}: {model.p[(i,b)].value}')  
        if model.p[(i,b)].value >= 0.99:
            I_b[b].append(i)
            
XY_pos = {}
for i in model.Items:
    XY_pos[i]={'x':int(model.x[i].value),'y':int(model.y[i].value),
               'L':int(I[i]['L']*model.r[i,1,1].value+I[i]['W']*model.r[i,1,2].value),
               'W':int(I[i]['L']*model.r[i,2,1].value+I[i]['W']*model.r[i,2,2].value)}
            
#########################
### Plotting solution ###
#########################
from matplotlib.patches import Rectangle
axis_font  = {'fontname':'Arial', 'size':'15'}

random.seed(42)

plt.close('all')
for b in B.keys():
    if len(I_b[b])>0:
        fig, ax = plt.subplots()
        for i in I_b[b]:
            ax.add_patch(Rectangle((XY_pos[i]['x'],
                                    XY_pos[i]['y']),
                                   XY_pos[i]['L'],XY_pos[i]['W'],
                 edgecolor = 'green',
                 facecolor =  [random.randint(0,255)/255, 
                               random.randint(0,255)/255, 
                               random.randint(0,255)/255 ],
                 fill=True,
                 lw=1))
            plt.text((XY_pos[i]['x']+XY_pos[i]['x']+XY_pos[i]['L'])/2,
                     (XY_pos[i]['y']+XY_pos[i]['y']+XY_pos[i]['W'])/2,
                     str(i),fontsize=15,color='w')
        ax.set_xlim(0,B[b]['L'])
        ax.set_ylim(0,B[b]['W'])
        ax.set_xticks(range(0,B[b]['L']+1))
        ax.set_yticks(range(0,B[b]['W']+1))
        ax.set_xlabel('Length',**axis_font)
        ax.set_ylabel('Width',**axis_font)
        ax.grid(True)
        plt.show()
        fig.savefig('bin_%i.png'%(b), format='png', dpi=400, bbox_inches='tight',
                 transparent=True,pad_inches=0.02)    



