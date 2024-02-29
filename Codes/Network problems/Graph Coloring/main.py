# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 21:16:30 2024

@author: abombelli
"""

import numpy as np
import pandas as pd
import os
import igraph as ig
import networkx as nx
import random
from gurobipy import Model,GRB,LinExpr,quicksum
import cartopy
import cartopy.crs as ccrs
import matplotlib as mp
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.patheffects as PathEffects
import cartopy.io.shapereader as shpreader
import cartopy.feature as cfeature
from pyomo.environ import *


random.seed(7)
plt.close('all')

with open('adjacent-states.txt') as f: # lines like 'CA,OR,NV,AZ'
    adjlist = [line.strip().split(',') for line in f
                                       if line and line[0].isalpha()]
    
adj_dict     = {v[0]:v[1:] for v in adjlist}

idx_to_state_dict = {idx:k for idx,k in enumerate(adj_dict.keys())}
state_to_idx_dict = {v:k for k,v in idx_to_state_dict.items()}

V = [k for k in idx_to_state_dict.keys()]

E = []
for state1 in adj_dict.keys():
    for state2 in adj_dict[state1]:
        if (state_to_idx_dict[state2],state_to_idx_dict[state1]) not in E:
            E.append((state_to_idx_dict[state1],state_to_idx_dict[state2]))

###############################
### Plotting continental US ###
###############################
            
scale     = '110m'
states110 = cfeature.NaturalEarthFeature(
            category='cultural',
            name='admin_1_states_provinces_lines',
            scale=scale,
            facecolor='none',
            edgecolor='k')


            
crs = ccrs.PlateCarree()
fig, ax = plt.subplots(
    1, 1, 
    figsize=(15, 10),
    subplot_kw=dict(projection=crs))
ax.coastlines()
ax.add_feature(states110, zorder=1, linewidth=1.0)
ax.add_feature(cartopy.feature.LAND,linewidth=0.1,facecolor='w',alpha=0.9)
ax.add_feature(cartopy.feature.OCEAN,facecolor=(242./255,242./255,242./255))
ax.add_feature(cartopy.feature.COASTLINE,linewidth=0.25)
ax.add_feature(cartopy.feature.BORDERS,linestyle='-',linewidth=1.5,edgecolor='k')

ax.set_extent([-128,-60,20,50],
                ccrs.PlateCarree())
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=2, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = True
gl.left_labels = True
gl.xlines = True
gl.xlines = True
gl.xlocator = mticker.FixedLocator([-160,-140,-120,-100,-80,-60,-40,-20,0,20,40,60,80,100,120,140,160])
gl.ylocator = mticker.FixedLocator([-65,-45,-25,25,35,45,55])
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabel_style = {'size': 15, 'color': 'gray', 'weight': 'normal'}
gl.ylabel_style = {'size': 15, 'color': 'gray', 'weight': 'normal'} 

#%%

##########################
### Defining the graph ###       
##########################     
G        = nx.Graph()
for v in V:
    G.add_node(v,name=idx_to_state_dict[v])
for e in E:
    G.add_edge(e[0],e[1])

#######################################
### Defining the optimization model ###
#######################################    
model          = ConcreteModel()
model.vertices = Set(initialize=V)
model.edges    = Set(initialize=E)
K              = len(model.vertices)
model.colors   = Set(initialize=list(range(0,K)))

# Define the decision variables
model.x = Var(model.vertices, model.colors, domain=Binary)
model.y = Var(model.colors, domain=Binary)

# Define the objective function
model.obj = Objective(expr=sum(model.y[c] for c in model.colors), sense=minimize)

# Define the constraints
model.assign_color = ConstraintList()
for v in model.vertices:
    model.assign_color.add(quicksum(model.x[(v,c)] for c in model.colors)==1)
    
model.activate_color = ConstraintList()
for v in model.vertices:
    for c in model.colors:
        model.activate_color.add(model.x[v,c] <= model.y[c])
        
model.adjacent_vertices = ConstraintList()
for e in model.edges:
    for c in model.colors:
        model.adjacent_vertices.add(model.x[e[0],c]+model.x[e[1],c] <= 1)

consecutive_colors = True        

if consecutive_colors:

    model.consecutive_colors = ConstraintList()
    for c in model.colors:
        if c != 0:
            model.consecutive_colors.add(model.y[c]<=model.y[c-1]) 
        
# Solve the problem
solver = SolverFactory('gurobi')
solver.solve(model)

# Store assigned colors
v_c_dict = {}
for v in model.vertices:
    for c in model.colors:
        if model.x[(v,c)].value >= 0.99:
            v_c_dict[v] = c

###########################
### Analysis of results ###
###########################


print('')
print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
print('The chromatic color is %i'%(model.obj()))
print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
print('')

#%%
# Print the states with the same color
idx_colors_used    = list(np.unique(list(v_c_dict.values())))

colors_states_dict = {c:[idx_to_state_dict[k] for k,v in v_c_dict.items() if v==c] for c in idx_colors_used}



#%%

# get the data
fn = shpreader.natural_earth(
    resolution='10m', category='cultural', 
    name='admin_1_states_provinces')


reader      = shpreader.Reader(fn)
states      = [x for x in reader.records() if x.attributes["admin"] == "United States of America"]
states_geom = cfeature.ShapelyFeature([x.geometry for x in states], ccrs.PlateCarree())

data_proj = ccrs.PlateCarree()

# create the plot
fig, ax = plt.subplots(
    figsize=(15,10), dpi=130, facecolor="w",
    subplot_kw=dict(projection=data_proj),
)

ax.coastlines()
ax.add_feature(states110, zorder=1, linewidth=1.0)
ax.add_feature(cartopy.feature.LAND,linewidth=0.1,facecolor='w',alpha=0.9)
ax.add_feature(cartopy.feature.OCEAN,facecolor=(242./255,242./255,242./255))
ax.add_feature(cartopy.feature.COASTLINE,linewidth=0.25)
ax.add_feature(cartopy.feature.LAKES,zorder=2,linewidth=0.25)
ax.add_feature(cartopy.feature.BORDERS,linestyle='-',linewidth=1.5,edgecolor='k')
ax.add_feature(cfeature.BORDERS, color="k", lw=0.1)
ax.set_extent([-128,-60,20,50], crs=data_proj)
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=2, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = True
gl.left_labels = True
gl.xlines = True
gl.xlines = True
gl.xlocator = mticker.FixedLocator([-160,-140,-120,-100,-80,-60,-40,-20,0,20,40,60,80,100,120,140,160])
gl.ylocator = mticker.FixedLocator([-65,-45,-25,25,35,45,55])
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabel_style = {'size': 15, 'color': 'gray', 'weight': 'normal'}
gl.ylabel_style = {'size': 15, 'color': 'gray', 'weight': 'normal'}


if consecutive_colors:
    colors =[[205/255.0,133/255.0,63/255.0],
         [65/255.0,105/255.0,225/255.0],
         [34/255.0,139/255.0,34/255.0],
         [218/255.0,165/255.0,32/255.0],
         [205/255.0,133/255.0,63/255.0]]
else:
    colors          = [(random.randint(0,255)/255.0,random.randint(0,255)/255.0,
                        random.randint(0,255)/255.0) for k in range(0,K)]


ax.add_feature(states_geom, facecolor='none', edgecolor="k")


# add the names
for state in states:
    
    # For plotting purposes we exclude Alaska and Hawaii
    if state.attributes['postal'] != 'AK' and state.attributes['postal'] != 'HI': 
    
        lon  = state.geometry.centroid.x
        lat  = state.geometry.centroid.y
        name = state.attributes["name"] 
        
        ax.text(
            lon, lat, name, c='w',
            size=9, transform=data_proj, ha="center", va="center",
            path_effects=[PathEffects.withStroke(linewidth=7,
                                                 foreground=colors[v_c_dict[state_to_idx_dict[state.attributes['postal']]]])]
        )  

fig.savefig('graph_coloring_US.png', format='png', dpi=400, bbox_inches='tight',
             transparent=True,pad_inches=0.02) 

#%%

# create the plot
fig, ax = plt.subplots(
    figsize=(15,10), dpi=130, facecolor="w",
    subplot_kw=dict(projection=data_proj),
)

ax.coastlines()
ax.add_feature(states110, zorder=1, linewidth=1.0)
ax.add_feature(cartopy.feature.LAND,linewidth=0.1,facecolor='w',alpha=0.9)
ax.add_feature(cartopy.feature.OCEAN,facecolor=(242./255,242./255,242./255))
ax.add_feature(cartopy.feature.COASTLINE,linewidth=0.25)
ax.add_feature(cartopy.feature.LAKES,zorder=2,linewidth=0.25)
ax.add_feature(cartopy.feature.BORDERS,linestyle='-',linewidth=1.5,edgecolor='k')
ax.add_feature(cfeature.BORDERS, color="k", lw=0.1)
ax.set_extent([-80,-60,20,50], crs=data_proj)
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=2, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = True
gl.left_labels = True
gl.xlines = True
gl.xlines = True
gl.xlocator = mticker.FixedLocator([-160,-140,-120,-100,-80,-60,-40,-20,0,20,40,60,80,100,120,140,160])
gl.ylocator = mticker.FixedLocator([-65,-45,-25,25,35,45,55])
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabel_style = {'size': 15, 'color': 'gray', 'weight': 'normal'}
gl.ylabel_style = {'size': 15, 'color': 'gray', 'weight': 'normal'}


if consecutive_colors:
    colors =[[205/255.0,133/255.0,63/255.0],
         [65/255.0,105/255.0,225/255.0],
         [34/255.0,139/255.0,34/255.0],
         [218/255.0,165/255.0,32/255.0],
         [205/255.0,133/255.0,63/255.0]]
else:
    colors          = [(random.randint(0,255)/255.0,random.randint(0,255)/255.0,
                        random.randint(0,255)/255.0) for k in range(0,K)]


ax.add_feature(states_geom, facecolor='none', edgecolor="k")


# add the names
for state in states:
    
    # For plotting purposes we exclude Alaska and Hawaii
    if state.attributes['postal'] != 'AK' and state.attributes['postal'] != 'HI': 
    
        lon  = state.geometry.centroid.x
        lat  = state.geometry.centroid.y
        name = state.attributes["name"] 
        
        ax.text(
            lon, lat, name, c='w',
            size=9, transform=data_proj, ha="center", va="center",
            path_effects=[PathEffects.withStroke(linewidth=7,
                                                  foreground=colors[v_c_dict[state_to_idx_dict[state.attributes['postal']]]])]
        )  

#%%

######################
### Plotting the graph
######################

lat_lon_centroid_states = {}
for state in states:
    lon  = state.geometry.centroid.x
    lat  = state.geometry.centroid.y
    postal_code = state.attributes['postal']
    lat_lon_centroid_states[postal_code]=(lat,lon)
    

G_plot        = nx.Graph()
G_plot.pos    = {}
for v in V:
    G_plot.add_node(v,name=idx_to_state_dict[v])
    G_plot.pos[v] = (lat_lon_centroid_states[idx_to_state_dict[v]][1],
                     lat_lon_centroid_states[idx_to_state_dict[v]][0])
for e in E:
    G_plot.add_edge(e[0],e[1])
    
labels = {}
for v in V:
    labels[v] = idx_to_state_dict[v]
    
edge_alpha = 0.2
edge_color = [255/255.0,165/255.0,0/255.0]
font_color = [128/255.0,0/255.0,0/255.0]
    
fig, ax = plt.subplots(
    figsize=(15, 10))    
nx.draw_networkx(G_plot, ax=ax,
             font_size=13,
             font_color = font_color,
             alpha=0.8,
             width=1,
             node_size=1,
             pos=G_plot.pos,
             node_color='k',
             labels=labels,
             edge_color=edge_color)
lon_min = min([v[1] for v in lat_lon_centroid_states.values()])
lon_max = max([v[1] for v in lat_lon_centroid_states.values()])
lat_min = min([v[0] for v in lat_lon_centroid_states.values()])
lat_max = max([v[0] for v in lat_lon_centroid_states.values()])
ax.set_xlabel('Longitude (deg)',fontsize=20,font='Arial')
ax.set_ylabel('Latitude (deg)',fontsize=20,font='Arial')
ax.set_xlim(1.05*lon_min,0.95*lon_max)
ax.set_ylim(0.95*lat_min,1.05*lat_max)
ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
plt.show()
fig.savefig('graph_coloring_US_initial_graph.png', format='png', dpi=400, bbox_inches='tight',
             transparent=True,pad_inches=0.02) 