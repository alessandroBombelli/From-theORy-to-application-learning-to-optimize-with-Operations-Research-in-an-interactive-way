# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 13:56:11 2024

@author: abombelli
"""

import numpy as np
import pandas as pd
import os
import igraph as ig
from geopy.distance import great_circle
import random
from itertools import product
from gurobipy import Model,GRB,LinExpr,quicksum
from operator import itemgetter
import itertools

random.seed(1)

cwd = os.getcwd()


# Parameters
Life = 16

# Set of monsters
O      = {1:{'name':'Zorgoiln the Zombie','life_points':2,'coins':5},
          2:{'name':'Henry the Hermit Crab','life_points':5,'coins':17},
          3:{'name':'Ghost of your past','life_points':4,'coins':15},
          4:{'name':'Marion of the Haron','life_points':5,'coins':19},
          5:{'name':'Gerald the Gunk','life_points':14,'coins':55},
          6:{'name':'The Big Brown Bear','life_points':2,'coins':8},
          7:{'name':'The Frog Prince','life_points':2,'coins':8},
          8:{'name':'The Mummy','life_points':7,'coins':32}}






O_idx_first_stage  = [1,2,3,4]
O_idx_second_stage = sorted(list(set(O.keys())-set(O_idx_first_stage)))

O_first_stage  = {k:v for k,v in O.items() if k in O_idx_first_stage}
O_second_stage = {k:v for k,v in O.items() if k in O_idx_second_stage}

K            = 2
temp         = list(itertools.combinations(list(O_second_stage.keys()),K))
combinations = list(set(temp))
Prob         = 1/len(combinations)

S            = {k+1:c for k,c in enumerate(combinations)}

# Setup model
model = Model()

##########################
### Decision variables ###
##########################

x1 = {} # knapsack decision variables of first stage
x2 = {} # knapsack decision variables of second stage

for o in O_first_stage.keys():
    x1[o] = model.addVar(lb=0, ub=1, vtype=GRB.BINARY, name="x1[%s]"%(o))

for s in S.keys():
    for o in S[s]:
        x2[s,o] = model.addVar(lb=0, ub=1, vtype=GRB.BINARY, name="x2[%s,%s]"%(s,o))
        
C1 = model.addConstrs((quicksum(O_first_stage[o]['life_points']*x1[o] for o in O_first_stage.keys()) + 
                      quicksum(O_second_stage[o]['life_points']*x2[s,o] for o in S[s]) <= Life-1 for
                      s in S.keys()), 
                      name='C1')

##########################
### Objective function ###
##########################

obj = LinExpr()

# Contribution from first stage
for o in O_first_stage.keys():
    obj += O_first_stage[o]['coins']*x1[o]

# Contribution from second stage
for s in S.keys():
    for o in S[s]:
        obj += Prob*O_second_stage[o]['coins']*x2[s,o]
        
model.setObjective(obj,GRB.MAXIMIZE)
model.update()
model.write('third_level.lp')  
model.setParam('MIPGap',0.001)
model.setParam('TimeLimit',2*3600) # seconds
model.setParam('LogFile','third_level.log')
model.optimize() 

# Retrieve variable names and values
solution_stoc_model = []
for v in model.getVars():
    solution_stoc_model.append([v.varName,v.x])

# Retrieve active routing variables
active_variables_stoc_model = []
for i in range(0,len(solution_stoc_model)):
    if solution_stoc_model[i][1] >= 0.99:
        active_variables_stoc_model.append([solution_stoc_model[i][0],solution_stoc_model[i][1]])
        
print('')
print('Stochastic solution is %.1f'%(model.objVal))

solution_vec = []
idx_chosen   = []
act_var_vec  = []

for comb in combinations:
    idx_simulation     = list(O_first_stage.keys())
    for add_o in comb:
        idx_simulation.append(add_o)
    
    idx_simulation = sorted(idx_simulation)
    
    O_simulation   = {k:v for k,v in O.items() if k in idx_simulation}
    
    
    # Setup model
    model = Model()
    
    ##########################
    ### Decision variables ###
    ##########################
    
    x1 = {} # knapsack decision variables of first stage
    
    for o in O_simulation.keys():
        x1[o] = model.addVar(lb=0, ub=1, vtype=GRB.BINARY, name='x1[%s]'%(o))
    
    ###################
    ### Constraints ###
    ###################
    
    # In the first stage, we cannot defeat more monsters than our life 
    # points minus 1
    C1 = model.addConstr((quicksum(O_simulation[o]['life_points']*x1[o] for o
                                    in O_simulation.keys())<= Life-1), 
                      name="C1")
    
    ##########################
    ### Objective function ###
    ##########################
    
    obj = LinExpr()
    
    # Contribution from first stage
    for o in O_simulation.keys():
        obj += O_simulation[o]['coins']*x1[o]
        
    model.setObjective(obj,GRB.MAXIMIZE)
    model.update()
    model.Params.LogToConsole = 0
    model.optimize()
    
    solution_vec.append(model.objVal)
    idx_chosen.append(comb)
    
    # Retrieve variable names and values
    solution = []
    for v in model.getVars():
        solution.append([v.varName,v.x])
    
    # Retrieve active routing variables
    active_variables = []
    for i in range(0,len(solution)):
        if solution[i][1] >= 0.99:
            active_variables.append([solution[i][0],solution[i][1]])
            
    idx_defeated_opp = []
    for act_var in active_variables:
        idx_brck1 = [pos for pos,char in enumerate(act_var[0]) if char=='['][0]
        idx_brck2 = [pos for pos,char in enumerate(act_var[0]) if char==']'][0]
        idx_defeated_opp.append(int(act_var[0][idx_brck1+1:idx_brck2]))
    
    act_var_vec.append(idx_defeated_opp)
    
# Printing the average to show it is higher than the stochastic solution due to
# the Expected Value of Perfect Information (EVPI)

print('')
print('Solution with EVPI is %.1f'%(np.mean(solution_vec)))
        