# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 12:33:49 2022

@author: abombelli
"""

import numpy as np
import os
import pandas as pd
import time
import random
import pickle
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import PIL
# Import PuLP modeler functions
from pulp import *

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

# Create the 'prob' variable to contain the problem data
prob = LpProblem("Shortest_path", LpMinimize)

x_ij_vars = LpVariable.dicts("x",[(E[key][0],E[key][1]) for key in E.keys()],0,1)

# The objective function is added to 'prob' first
prob += (
    lpSum([E[key][3] * x_ij_vars[E[key][0],E[key][1]] for idx,key in enumerate(E.keys())]),
    "Total_distance",
)

print('Creating constraints')

for n in N.keys():
    if n == source:
        prob += (
            lpSum(x_ij_vars[(n,n_out)] for n_out in N_out[n]) -
            lpSum(x_ij_vars[(n_in,n)] for n_in in N_in[n]) == 1,
            "Source",)
    elif n == sink:
        prob += (
            lpSum(x_ij_vars[(n,n_out)] for n_out in N_out[n]) -
            lpSum(x_ij_vars[(n_in,n)] for n_in in N_in[n]) == -1,
            "Sink",)
    else:
        prob += (
            lpSum(x_ij_vars[(n,n_out)] for n_out in N_out[n]) -
            lpSum(x_ij_vars[(n_in,n)] for n_in in N_in[n]) == 0,
            "node_%s"%(n),)
 
# The problem data is written to an .lp file
prob.writeLP("SP.lp")

# The problem is solved using PuLP's choice of Solver
solver = getSolver('GLPK_CMD')
prob.solve()

# The status of the solution is printed to the screen
print("Status:", LpStatus[prob.status])

# Each of the variables is printed with it's resolved optimum value
for v in prob.variables():
    print(v.name, "=", v.varValue)
    
# The optimised objective function value is printed to the screen
print("Length path = ", value(prob.objective))

