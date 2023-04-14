# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 12:15:50 2023

@author: abombelli
"""

import numpy as np
import random
import os
import matplotlib.pyplot as plt
import pulp
from pulp import *

cwd = os.getcwd()
plt.close('all')

random.seed(8)

model = LpProblem("01_KP", LpMaximize)

N_B       = 2
B         = np.arange(0,N_B)
S_bar_min = 50
S_bar_max = 80
S_bar     = {b:np.round(S_bar_min+(S_bar_max-S_bar_min)*random.random(),0) for b in B}
N_I       = 10
I         = np.arange(0,N_I)
S_min     = 0.1
S_max     = 0.5 
S         = {i:np.round((S_min+(S_max-S_min)*random.random())*min(S_bar.values()),0) for i in I}
C_min     = 5
C_max     = 15 
V         = {i:np.round((C_min+(C_max-C_min)*random.random()),0) for i in I}



##########################
### Decision variables ###
##########################
x = LpVariable.dicts("x",(I,B),0,1,LpBinary)

##########################
### Objective function ###
##########################
model += lpSum(lpSum(V[i]*x[i][b] for i in I) for b in B)

###################
### Constraints ###
###################

# Each bin can be used up to its capacity
for b in B:
    model += (lpSum(S[i]*x[i][b] for i in I) <= S_bar[b],"bin_%s"%(b),)
    
# Each item can only be assigned to maximum one bin
for i in I:
    model += (lpSum(x[i][b] for b in B) <= 1,"item_%s"%(i),)

# optimizing

# The problem data is written to an .lp file
model.writeLP("01_KP.lp")

# Make sure to run the pulpTestAll() command to see which solvers are
# available/unavailable
solver = getSolver("PULP_CBC_CMD")

model.solve(solver)
print ("Status:", LpStatus[model.status])
for v in model.variables():
    print (v.name, "=", v.varValue)
print ("Optimal Solution = ", value(model.objective))

I_to_B = {b: [] for b in B}
for v in model.variables():
    if v.varValue >= 0.99:
        idx_openBr   = [pos for pos, char in enumerate(v.name) if char == '_' ]
        item_idx     = int(v.name[idx_openBr[0]+1:idx_openBr[1]])
        bin_idx      = int(v.name[idx_openBr[1]+1:])
        I_to_B[bin_idx].append(item_idx)
        
        
#%%
#######################
### Post-processing ###
#######################

# Recap solution:
for b in B:
    capacity_used   = sum([S[item] for item in I_to_B[b]])
    value_contained = sum([V[item] for item in I_to_B[b]]) 
    print('Bin %i (max. capacity = %.1f): used capacity = %.1f, value contained = %.1f'%(b,S_bar[b],capacity_used,value_contained))
    
    

plt.close('all')
axis_font  = {'fontname':'Arial', 'size':'15'}

#########################
### Plotting solution ###
#########################
from matplotlib.patches import Rectangle
vert_size_bin = 0.5


fig, ax = plt.subplots()
for b in B:
    ax.add_patch(Rectangle((0,
                            b-vert_size_bin/2),
                            S_bar[b],vert_size_bin,
          edgecolor = 'black',
          facecolor =  [random.randint(0,255)/255, 
                        random.randint(0,255)/255, 
                        random.randint(0,255)/255 ],
          fill=False,
          lw=1))
    x_pos = 0
    for item in I_to_B[b]:
        ax.add_patch(Rectangle((x_pos,
                            b-vert_size_bin/2),
                            S[item],vert_size_bin,
          edgecolor = 'black',
          facecolor =  [random.randint(0,255)/255, 
                        random.randint(0,255)/255, 
                        random.randint(0,255)/255 ],
          fill=True,
          lw=1))
        plt.text(x_pos+0.05*S[item],
                 b-0.5*vert_size_bin/2,
                 "(%i,%.1f)"%(item,S[item]),fontsize=6,color="white",weight="bold")
        # Update x position to plot new item
        x_pos += S[item]
    

ax.set_xlim(0,max(S_bar.values()))
ax.set_ylim(-vert_size_bin/2,len(B)-vert_size_bin/2)
ax.set_yticks(range(0,len(B)))
ax.set_xlabel('Capacity',**axis_font)
ax.set_ylabel('Bin',**axis_font)
ax.grid(True)
plt.show()
fig.savefig('01_KP_solution.png', format='png', dpi=400, bbox_inches='tight',
            transparent=True,pad_inches=0.02)



