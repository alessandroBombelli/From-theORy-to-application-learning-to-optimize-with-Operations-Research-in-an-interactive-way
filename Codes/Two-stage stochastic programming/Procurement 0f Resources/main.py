# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 18:19:43 2024

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

########################
### Stochastic model ###
########################

# Set of products
P = {1:{1:3,2:2,3:4,'revenue':25},
     2:{1:1,2:2,3:2,'revenue':20},
     3:{1:4,2:2,3:0,'revenue':10}}

# Set of resources
R = {1:{1:3,2:1,3:4,'cost':3},
     2:{1:2,2:2,3:2,'cost':2},
     3:{1:4,2:2,3:0,'cost':1}}

# Set of scenarios
S = {1:{1:40,2:30,3:10,'prob':0.5},
     2:{1:30,2:20,3:0,'prob':0.4},
     3:{1:10,2:30,3:50,'prob':0.1}}

# Define optimization model
model = ConcreteModel()

# Define sets
model.Products     = Set(initialize=P.keys())
model.Resources    = Set(initialize=R.keys())
model.Scenarios    = Set(initialize=S.keys())

# Define parameters

Q_rp = {(r,p):v[p] for r,v in R.items() for p in P.keys()}

model.Q_rp = Param(model.Resources,model.Products, initialize=
             {(r,p):v[p] for r,v in R.items() for p in P.keys()}, within=Any)
model.R_p  = Param(model.Products, initialize={p:v['revenue'] for p,v in P.items()})
model.C_r  = Param(model.Resources, initialize={r:v['cost'] for r,v in R.items()})
model.D_ps = Param(model.Products,model.Scenarios, initialize=
             {(p,s):v[p] for p in P.keys() for s,v in S.items()}, within=Any)
model.P_s  = Param(model.Scenarios, initialize={s:v['prob'] for s,v in S.items()})   

# Define decision variables
model.x = Var(model.Resources, within=NonNegativeReals)
model.y = Var(model.Products,model.Scenarios, within=NonNegativeReals)

# Define objective function
model.obj = Objective(expr=-sum(model.C_r[r]*model.x[r] for r in model.Resources)
                      +sum(model.P_s[s]*(sum(model.R_p[p]*model.y[p,s] for p in model.Products)) 
                           for s in model.Scenarios), sense=maximize)

# Define constraints
model.max_products_per_resource_scenario = ConstraintList()
for s in model.Scenarios:
    for r in model.Resources:
        model.max_products_per_resource_scenario.add(
                sum(model.Q_rp[r,p]*model.y[p,s] for p in model.Products)<=model.x[r])
        
model.max_products_demand_scenario = ConstraintList()
for s in model.Scenarios:
    for p in model.Products:
        model.max_products_demand_scenario.add(model.y[p,s]<=model.D_ps[p,s])
        
# Solve the problem
solver = SolverFactory('gurobi')
solver.solve(model)  

# Print the results
print('Expected profit with stochastic solution:', model.obj())
print('')

for r in model.Resources:
    print(f'Resource {r}: {model.x[(r)].value}') 
    
for s in model.Scenarios:
    print('Scenario %i:'%(s))
    for p in model.Products:
        print(f'Product {p}: {model.y[(p,s)].value}') 
    print('')
    

        