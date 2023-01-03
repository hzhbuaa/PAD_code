#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 15:52:38 2023

@author: hzh
"""

import numpy as np
import random
import networkx as nx
import pickle
from multiprocessing.pool import Pool


def creat_pow_law(k,bound,n):
    A = (1-k)/(1-bound**(-k+1))
    n_array = np.random.rand(n)
    return np.exp(-np.log((A*bound**(-k+1)-n_array*k+n_array)/A)/(k-1))

def creat_constant(key,n):
    array = key*np.ones(n)
    return array,key,key*key

class simplex_vertex:

    def __init__(self, vertex_id, activity, probability):
        self.act = activity
        self.p = probability
        self.name = vertex_id
        self.memory_one = []
        self.memory_high = []
    
    def new_edge(self, node, size):
        nodes = list(np.array(node))
        nodes.remove(self.name)
        edge_list = []
        choice_nodes = random.sample(nodes, size)
        for num in range(len(choice_nodes)):
            edge_list.append((self.name, choice_nodes[num]))
        self.memory_one.append(edge_list)
        return edge_list
    
    def new_collab(self, node, size):
        nodes = list(np.array(node))
        nodes.remove(self.name)
        e1 = random.sample(nodes,size-1)
        e1.append(self.name)
        self.memory_high.append(e1)
        return e1

def add_clique(e, tgraph):
    g = nx.complete_graph(len(e))
    rl = dict(zip(range(len(e)), e))
    g = nx.relabel_nodes(g, rl)
    tgraph.add_edges_from(g.edges())
    return tgraph

def creat_random_size(pro_array):
    rand_num = np.random.rand()
    for i in range(len(pro_array)):
        if rand_num<=pro_array[i]:
            return i

def memoryless_instant_graph(vertex_dict, s_list, s_probability, m_list):
    tgraph = nx.Graph()
    tgraph.add_nodes_from(list(vertex_dict.keys()));
    new_history_one = []
    new_history_high = []
    nodes = list(vertex_dict.keys())
    for n in vertex_dict:
        if np.random.rand() <= vertex_dict[n].act:
            rand_index = creat_random_size(s_probability)
            s, m = s_list[rand_index], m_list[rand_index]
            if np.random.rand() <= vertex_dict[n].p:
                e_one = vertex_dict[n].new_edge(nodes, m)
                tgraph.add_edges_from(e_one)
                for num in range(len(e_one)):
                    new_history_one.append(e_one[num])
            else:
                e_high = vertex_dict[n].new_collab(nodes,s)
                new_history_high.append(e_high);
                tgraph = add_clique(e_high,tgraph);
    return tgraph, new_history_one, new_history_high;

def temporal_graph_creation(N, T, s_list, s_probability, m_list, act, p_list, seed=1, verbose=True):
    tgraph = {}
    vertex_dict = {}
    edge = {}
    simplex = {}
    for n in range(N):
        vertex_dict[n] = simplex_vertex(n, act[n], p_list[n])
    for t in range(T):
        tg, edge_list, tri_list = memoryless_instant_graph(vertex_dict, s_list, s_probability, m_list)
        tgraph[t] = tg
        edge[t] = edge_list
        simplex[t] = tri_list
    return tgraph, vertex_dict, edge, simplex

def aggregate_graph(TG):
    w = {}
    for t in TG:
        edges = TG[t].edges();
        for edge in edges:
            if edge not in w:
                w[edge] = 1;
            #w[edge]+=1;
    G = nx.Graph();
    G.add_nodes_from(list(TG[0].nodes()));
    G.add_edges_from(w.keys());
    nx.set_edge_attributes(G,w,'weight');
    return G;

def creat_one_network(arg):
    N, T, act, a_value, p_value, s_list, s_probability,m_list = arg
    p_list, p, p1 = creat_constant(p_value, N)
    TG, vertex_dict, edge, simplex = temporal_graph_creation(N, T, s_list, s_probability, m_list, act, p_list, seed=10)
    dicts = dict(zip(['act_value','p_value','s_list','s_probability','m_list','N','T','TG','vertex_dict','edge','simplex'],[a_value,p_value,s_list,s_probability,m_list,N,T,TG,vertex_dict,edge,simplex]))
    savename = './new_psad_constant_N={}_T={}_a={}_p={}_slist={}_sprobability={}_mlist={}.data'.format(N,T,a_value,p_value,s_list,s_probability,m_list)
    with open(savename,'wb') as file:
        pickle.dump(dicts,file)

if __name__=='__main__':
    N = 1500
    a_value = 0.1
    p_list = [0.55]
    act,a,a1 = creat_constant(a_value, N)
    s_list = [3,4,5]
    s_probability = [1/3,2/3,1]
    m_list = [0]*len(s_list)
    for i in range(len(s_list)):
        s = s_list[i]
        m_list[i] = int(s*(s-1)/2)
    T = 100
    args = []
    for p_value in p_list:
        args.append([N, T, act, a_value, p_value, s_list, s_probability, m_list])
    pool = Pool(processes=1)
    pool.map(creat_one_network, args)