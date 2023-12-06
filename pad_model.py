#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 15:52:38 2023

@author: hanzhihao
"""

import numpy as np
import random
import networkx as nx
import pickle
from multiprocessing.pool import Pool
import time
import tqdm
import json
import math
import os


def creat_power_law(eta, epsilon, n):
    value = (1 - eta) / (1 - epsilon ** (-eta + 1))
    random_array = np.random.rand(n)
    array = np.exp(
        -np.log((value * epsilon ** (-eta + 1) - random_array * eta + random_array) / value) / (eta - 1))
    return array


def creat_constant(key, n):
    array = key * np.ones(n)
    return array


def creat_uniform(n):
    array = np.random.rand(n)
    return array


def calculate_cdf(pdf_array):
    total = 0
    cdf_array = []
    for value in pdf_array:
        total += value
        cdf_array.append(total)
    return cdf_array


class Simplex_Vertex:
    def __init__(self, vertex_id, activity, probability):
        self.act = activity
        self.p = probability
        self.name = vertex_id
        self.memory_one = []
        self.memory_high = []

    def new_edge(self, nodes, size):
        nodes.remove(self.name)
        edge_list = []
        choice_nodes = random.sample(nodes, size)
        for choice_node in choice_nodes:
            edge_list.append((self.name, choice_node))
        self.memory_one.append(edge_list)
        return edge_list

    def new_collab(self, nodes, size):
        nodes.remove(self.name)
        simplex = random.sample(nodes, size - 1)
        simplex.append(self.name)
        self.memory_high.append(simplex)
        return simplex


def add_clique(e, instant_network):
    g = nx.complete_graph(len(e))
    rl = dict(zip(range(len(e)), e))
    g = nx.relabel_nodes(g, rl)
    instant_network.add_edges_from(g.edges())
    return instant_network


def creat_random_size(pro_cdf):
    rand_num = np.random.rand()
    for index in range(len(pro_cdf)):
        if rand_num <= pro_cdf[index]:
            return index


def instant_network_generate(vertex_dict, simplex_size, pro_cdf, m_list):
    instant_network = nx.Graph()
    instant_network.add_nodes_from(list(vertex_dict.keys()))
    new_history_one = []
    new_history_high = []
    nodes = list(vertex_dict.keys())
    for n in vertex_dict:
        if np.random.rand() <= vertex_dict[n].act:
            rand_index = creat_random_size(pro_cdf)
            size, m = simplex_size[rand_index], m_list[rand_index]
            if np.random.rand() >= vertex_dict[n].p:
                edges = vertex_dict[n].new_edge(nodes, m)
                instant_network.add_edges_from(edges)
                for edge in edges:
                    new_history_one.append(edge)
            else:
                simplex = vertex_dict[n].new_collab(nodes, size)
                new_history_high.append(simplex)
                instant_network = add_clique(simplex, instant_network)
    return instant_network, new_history_one, new_history_high


def temporal_network_generate(size, step, simplex_size, pro_cdf, m, act, p, idx):
    time.sleep(idx/10)
    t_network = {}
    vertex_dict = {}
    edge = {}
    simplex = {}
    for n in range(size):
        vertex_dict[n] = Simplex_Vertex(n, act[n], p[n])
    pbar = tqdm.tqdm(range(step))
    for t in pbar:
        pbar.set_description('Processing %s' % str(idx))
        instant_network, instant_edge, instant_simplex = instant_network_generate(vertex_dict, simplex_size, pro_cdf, m)
        t_network[t] = instant_network
        edge[t] = instant_edge
        simplex[t] = instant_simplex
    return t_network, vertex_dict, edge, simplex


def aggregate_graph(TG):
    w = {}
    for t in TG:
        edges = TG[t].edges()
        for edge in edges:
            if edge not in w:
                w[edge] = 1
            # w[edge]+=1;
    G = nx.Graph()
    G.add_nodes_from(list(TG[0].nodes()))
    G.add_edges_from(w.keys())
    nx.set_edge_attributes(G, w, 'weight')
    return G


def generate_pad_network(parameters):
    size, step, act, p, simplex_size, pro_cdf, m, eta, epsilon, p_value, idx = parameters
    temporal_network, vertex_dict, edge, simplex = temporal_network_generate(size, step, simplex_size, pro_cdf, m, act, p, idx)
    dicts = dict(zip(['act_array', 'p_array', 'simplex_size', 'pro_cdf',
                      'm_list', 'N', 'T', 'temporal_network', 'vertex_dict', 'edge', 'simplex', 'eta', 'epsilon', 'p_value'],
                     [act, p, simplex_size, pro_cdf, m, size, step, temporal_network, vertex_dict, edge, simplex, eta, epsilon, p_value]))
    filename = 'simplex_size={}_pro_cdf={}_m={}_N={}_T={}_eta={}_epsilon={}_p={}.data'.format(simplex_size, pro_cdf, m, size, step, eta, epsilon, p_value)
    output_dir = 'network_data'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(os.path.join(output_dir, filename), 'wb') as file:
        pickle.dump(dicts, file)


if __name__ == '__main__':
    size, step, activity_mean = 1500, 40000, 0.1
    eta_epsilon_dict = {1.1: 0.00057233460371226, 2.3: 0.03609470062606364, 3.5: 0.060858890768500505}
    eta_list = [3.5]
    p_list = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    p_array_list = []
    for p in p_list:
        p_array_list.append(creat_constant(p, size))
    simplex_size = [3,4,5]
    pro_cdf = [1/3, 2/3, 1]
    m = []
    for z in simplex_size:
        m.append(int(z * (z - 1) / 2))
    paras = []
    act_array_all = []
    for i in range(len(eta_list)):
        error = 1
        use_activity = np.array([])
        epsilon = eta_epsilon_dict[eta_list[i]]
        for j in range(200):
            activity_array = creat_power_law(eta_list[i], epsilon, size)
            if abs(activity_array.mean()-activity_mean) < error:
                error = abs(activity_array.mean()-activity_mean)
                use_activity = activity_array.copy()
        act_array_all.append(use_activity)
    paras = []
    for i in range(len(p_array_list)):
        for j in range(len(act_array_all)):
            paras.append([size, step, act_array_all[j], p_array_list[i], simplex_size, pro_cdf, m, eta_list[j], eta_epsilon_dict[eta_list[j]], p_list[i]])
    random.shuffle(paras)
    for idx in range(len(paras)):
        paras[idx].append(idx)
    pool = Pool(processes=24)
    pool.map(generate_pad_network, paras)