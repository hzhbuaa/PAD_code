#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 22 16:52:39 2021

@author: hzh
"""

import numpy as np
import random
import networkx as nx
import pickle
from multiprocessing.pool import Pool
import tqdm
from itertools import combinations
import os
import copy

# In[8]:

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

class Variable_class:
    def __init__(self, edge, simplex, N):
        self.N = N
        self.edge_dict = {}
        self.triangles_dict = {}
        for t_value, simplex_list in tqdm.tqdm(simplex.items()):
            temporal_edge = set()
            temporal_triangle = set()
            for one_simplex in simplex_list:
                for triangle in combinations(one_simplex, 3):
                    temporal_triangle.add(tuple(sorted(triangle)))
                for one_edge in combinations(one_simplex, 2):
                    temporal_edge.add(tuple(sorted(one_edge)))
            for one_edge in edge[t_value]:
                temporal_edge.add(tuple(sorted(one_edge)))
            self.edge_dict[t_value] = list(temporal_edge)
            self.triangles_dict[t_value] = list(temporal_triangle)
    
        
class SISModel:
    def __init__(self, N, I_percentage):
        #self.edge_dict = edge_dict
        #self.triangles_dict = triangles_dict
        self.nodes = [i for i in range(N)]
        self.N = N
        self.I = I_percentage * self.N/100
        #self.initial_infected_nodes = self.initial_setup()
    
    def initial_setup(self,fixed_nodes_to_infect=None,print_status=True):
        self.sAgentSet = set()
        self.iAgentSet = set()
        
        self.iList = []
        self.i_mean_a = []
        self.t = 0
        
        for n in self.nodes:
            self.sAgentSet.add(n)
        if fixed_nodes_to_infect==None: 
            infected_this_setup=[]
            for ite in range(int(self.I)): 
                to_infect = random.choice(list(self.sAgentSet))
                self.infectAgent(to_infect)
                infected_this_setup.append(to_infect)
        else: 
            infected_this_setup=[]
            for to_infect in fixed_nodes_to_infect:
                self.infectAgent(to_infect)
                infected_this_setup.append(to_infect)
        #if print_status: print ('Setup:', self.N, 'nodes', self.I, 'infected')
        return len(infected_this_setup)
    
    def infectAgent(self,agent):
        self.iAgentSet.add(agent)
        self.sAgentSet.remove(agent)
        return 1
    def recoverAgent(self,agent):
        self.sAgentSet.add(agent)
        self.iAgentSet.remove(agent)
        return -1
    
    def run(self, beta, beta2, mu, print_status=True):
        self.t_max = len(variable.edge_dict)
        
        while len(self.iAgentSet) > 0 and len(self.sAgentSet) != 0 and self.t<self.t_max:
            newIlist = set()
            
            for edges in variable.edge_dict[self.t]:
                begin,end = edges
                if (begin in self.iAgentSet) and (end in self.sAgentSet):
                    if (random.random() <= beta):
                        newIlist.add(end)
                if (begin in self.sAgentSet) and (end in self.iAgentSet):
                    if (random.random() <= beta):
                        newIlist.add(begin)
              
            #TRIANGLE CONTAGION
            
            for triangle_value in variable.triangles_dict[self.t]:
                n1,n2,n3 = triangle_value
                if n1 in self.iAgentSet:
                    if n2 in self.iAgentSet:
                        if n3 in self.sAgentSet:
                            #infect n3 with probability beta2
                            if (random.random() < beta2): 
                                newIlist.add(n3)
                    else:
                        if n3 in self.iAgentSet:
                            #infect n2 with probability beta2
                            if (random.random() < beta2): 
                                newIlist.add(n2)
                else:
                    if (n2 in self.iAgentSet) and (n3 in self.iAgentSet):
                        #infect n1 with probability beta2
                        if (random.random() < beta2): 
                            newIlist.add(n1)
            
            for n_to_infect in newIlist:
                self.infectAgent(n_to_infect)
            
            #for recoveries
            newSlist = set()
            
            if len(self.iAgentSet)<self.N:
                for recoverAgent in self.iAgentSet:
                    #if the agent has just been infected it will not recover this time
                    if recoverAgent in newIlist:
                        continue
                    else:
                        if (random.random() < mu): 
                            newSlist.add(recoverAgent)

            for n_to_recover in newSlist:
                self.recoverAgent(n_to_recover)
            
            self.iList.append(len(self.iAgentSet))
            
            #increment the time
            self.t += 1
        '''if print_status: 
            print('lambda', beta/mu, 'Done!', len(self.iAgentSet), 'infected agents left')'''
        return self.iList
    
    def get_stationary_rho(self, normed=True, last_k_values = 10):
        i = self.iList
        if len(i)==0:
            return 0
        if normed:
            i = 1.*np.array(i)/self.N
        #print(i)
        if i[-1]==1:
            return 1
        elif i[-1]==0:
            return 0
        else:
            avg_i = np.mean(i[-last_k_values:])
            avg_i = np.nan_to_num(avg_i) #if there are no infected left nan->0   
            return avg_i


def run_one_simulation(par):
    I_percentage, beta1, beta2, mu, it_num, idx = par
    mySISModel = SISModel(variable.N, I_percentage)
    frc_infected = []
    pbar = tqdm.tqdm(range(it_num))
    for repeat in pbar:
        pbar.set_description('Processing {}'.format(str(idx)))
        mySISModel.initial_setup(fixed_nodes_to_infect=None)
        i_lists = mySISModel.run(beta1, beta2, mu, print_status=True)
        rho = mySISModel.get_stationary_rho(normed=True, last_k_values=10)
        frc_infected.append(rho)
    dicts = dict(zip(['it_num', 'I_percentage', 'beta1', 'beta2', 'mu', 'lambda1', 'lambda2', 'infected'],[it_num, I_percentage, beta1, beta2, mu, beta1/mu, beta2/mu, frc_infected]))
    return dicts


if __name__ == '__main__':
    filename_list = ['simplex_size=[3, 4, 5]_pro_cdf=[0.3333333333333333, 0.6666666666666666, 1]_m=[3, 6, 10]_N=1500_T=40000_eta=2.3_epsilon=0.03608615000027534_p=0.5.data']
    I_percentage_list = np.array([3,70])
    mu = 0.001
    beta1_list = np.linspace(0,1.2,121)*mu
    beta2_list = np.linspace(0,8,201)*mu
    times = 20
    filename_idx = 0
    with open(os.path.join('network_data',filename_list[filename_idx]),'rb') as file:
        dicts = pickle.load(file)
    file.close()
    N = dicts['N']
    T = dicts['T']
    eta = dicts['eta']
    epsilon = dicts['epsilon']
    p_value = dicts['p_value']
    m = dicts['m_list']
    simplex_size = dicts['simplex_size']
    pro_cdf = dicts['pro_cdf']
    TG = dicts['temporal_network']
    edge = dicts['edge']
    simplex = dicts['simplex']
    variable = Variable_class(edge, simplex, N)
    paras = []
    for i_per in I_percentage_list:
        for beta1 in beta1_list:
            for beta2 in beta2_list:
                paras.append([i_per, beta1, beta2, mu, times])
    random.shuffle(paras)
    for idx in range(len(paras)):
        paras[idx].append(idx)
    pool = Pool(processes=24)
    res_return = []
    for para in paras:
        res_one = pool.apply_async(run_one_simulation, args=(para,))
        res_return.append(res_one)
    pool.close()
    pool.join()
    res_all = [res_one.get() for res_one in res_return]
    savename = ''
    with open(savename,'wb') as file:
        pickle.dump(res_all,file)