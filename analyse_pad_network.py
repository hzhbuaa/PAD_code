import numpy as np
import random
import networkx as nx
from collections import Counter
import pickle
import matplotlib.pyplot as plt
import tqdm
from itertools import combinations

def calculate_k2_mean(simplex, num_node):
    all_triangle = set()
    for t_value, simplex_list in tqdm.tqdm(simplex.items()):
        for one_simplex in simplex_list:
            for triangle in combinations(one_simplex, 3):
                all_triangle.add(tuple(sorted(triangle)))
    k2_degree_dict = {}
    for idx in range(num_node):
        k2_degree_dict[idx] = 0
    for triangle in all_triangle:
        for node in triangle:
            k2_degree_dict[node] += 1
    k2_mean = np.array(list(k2_degree_dict.values())).mean()
    return k2_mean

def analyse(G, simplex):
    dicts = {}
    dicts['num_node'] = len(G.nodes())
    dicts['k2_mean'] = calculate_k2_mean(simplex, len(G.nodes()))
    dicts['mean_cluster'] = nx.average_clustering(G)
    all_degree = dict(G.degree())
    degree_list = list(all_degree.values())
    mean_degree = np.array(degree_list).mean()
    dicts['k1_mean'] = mean_degree
    total = len(degree_list)
    degree_distribution = dict(Counter(degree_list))
    for key,value in degree_distribution.items():
        degree_distribution[key] = value/total
    degree_new = []
    for key,value in degree_distribution.items():
        degree_new.append([key,value])
    degree_new.sort(key=lambda x:x[0])
    x_draw = [x[0] for x in degree_new]
    y_draw = [y[1] for y in degree_new]
    dicts['x_draw'] = x_draw
    dicts['y_draw'] = y_draw
    return dicts