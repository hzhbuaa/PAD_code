#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 17:12:45 2023

@author: hzh
"""

import numpy as np
import random
import pickle
from multiprocessing.pool import Pool

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

class SISModel():
    def __init__(self, TG, I_percentage, edge, simplex, vertex_dict):
        self.edge_dict = {}
        self.triangles_dict = {}
        self.vertex_dict = vertex_dict
        for t_value,triangle in simplex.items():
            temp_edge = []
            temp_triangle = []
            for i in range(len(triangle)):
                nums = triangle[i]
                self.dfs(temp_edge,nums, 2,0,[])
                self.dfs(temp_triangle,nums, 3,0,[])
            for num in range(len(edge[t_value])):
                temp_edge.append(edge[t_value][num])
            self.edge_dict[t_value] = temp_edge
            self.triangles_dict[t_value] = temp_triangle
        self.neighbors = {}
        for i in range(len(TG)):
            neights = {}
            for j in TG[i].nodes():
                neights[j] = set(TG[i].neighbors(j))
            self.neighbors[i] = neights
        self.nodes = list(self.neighbors[0].keys())
        self.N = len(self.neighbors[0].keys())
        self.I = I_percentage * self.N/100
        #self.initial_infected_nodes = self.initial_setup()

    def dfs(self,ans,num_list,n,point,temp_list):
        if len(temp_list)==n:
            ans.append(tuple(temp_list[:]))
            return
        if len(num_list)<n or point>=len(num_list):
            return
        self.dfs(ans,num_list,n,point+1,temp_list)
        temp_list.append(num_list[point])
        self.dfs(ans,num_list,n,point+1,temp_list)
        temp_list.pop()
    
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
        self.t_max = len(self.edge_dict)
        
        while len(self.iAgentSet) > 0 and len(self.sAgentSet) != 0 and self.t<self.t_max:
            newIlist = set()
            
            for edges in self.edge_dict[self.t]:
                begin,end = edges
                if (begin in self.iAgentSet) and (end in self.sAgentSet):
                    if (random.random() <= beta):
                        newIlist.add(end)
                if (begin in self.sAgentSet) and (end in self.iAgentSet):
                    if (random.random() <= beta):
                        newIlist.add(begin)
              
            for triangle_value in self.triangles_dict[self.t]:
                n1,n2,n3 = triangle_value
                if n1 in self.iAgentSet:
                    if n2 in self.iAgentSet:
                        if n3 in self.sAgentSet:
                            if (random.random() < beta2): 
                                newIlist.add(n3)
                    else:
                        if n3 in self.iAgentSet:
                            if (random.random() < beta2): 
                                newIlist.add(n2)
                else:
                    if (n2 in self.iAgentSet) and (n3 in self.iAgentSet):
                        if (random.random() < beta2): 
                            newIlist.add(n1)
            
            for n_to_infect in newIlist:
                self.infectAgent(n_to_infect)
            
            newSlist = set()
            if len(self.iAgentSet)<self.N:
                for recoverAgent in self.iAgentSet:
                    if recoverAgent in newIlist:
                        continue
                    else:
                        if (random.random() < mu): 
                            newSlist.add(recoverAgent)

            for n_to_recover in newSlist:
                self.recoverAgent(n_to_recover)
            self.iList.append(len(self.iAgentSet))
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
        if i[-1]==1:
            return 1
        elif i[-1]==0:
            return 0
        else:
            avg_i = np.mean(i[-last_k_values:])
            avg_i = np.nan_to_num(avg_i)
            return avg_i


def run_one_simulation(par):
    mySISModel, I_percentage, beta, beta2, mu,it_num = par
    mySISModel.I = I_percentage * mySISModel.N / 100
   #mySISModel.initial_setup(fixed_nodes_to_infect=mySISModel.initial_infected_nodes)
    temps = []
    for defeat in range(it_num):
        mySISModel.initial_setup(fixed_nodes_to_infect=None)
        mySISModel.run(beta,beta2,mu,print_status=True)
        rho = mySISModel.get_stationary_rho(normed=True,last_k_values=8)
        temps.append(rho)
    frc_infected = temps
    dicts = dict(zip(['it_num','I_percentage','beta','beta2','mu','lambda1','lambda2','infected'],[it_num,I_percentage,beta,beta2,mu,beta/mu,beta2/mu,frc_infected]))
    return dicts

def run_one_network(arg):
    filename,beta_list,beta2_list,mu,I_percentage_list,times = arg
    with open(filename,'rb') as file:
        dicts = pickle.load(file)
    N = dicts['N']
    T = dicts['T']
    a_value = dicts['act_value']
    p_value = dicts['p_value']
    s_list = dicts['s_list']
    s_probability = dicts['s_probability']
    TG = dicts['TG']
    vertex_dict = dicts['vertex_dict']
    edge = dicts['edge']
    simplex = dicts['simplex']
    mySISModel = SISModel(TG,I_percentage_list[0],edge,simplex,vertex_dict)
    results = []
    for i_per in I_percentage_list:
        for beta_value in beta_list:
            for beta2_value in beta2_list:
                par = [mySISModel,i_per,beta_value,beta2_value,mu,times]
                results.append(run_one_simulation(par))
    savename = './lambda2_influence/lambda2_influence_N={}_T={}_a={}_p={}_slist={}_spro={}.txt'.format(N,T,a_value,p_value,s_list,s_probability)
    with open(savename,'wb') as file:
        pickle.dump(results,file)


if __name__ == '__main__':
    filename_list = []
    p_list = [0.5]
    name_model = ''
    for p_value in p_list:
        filename_list.append(name_model.format(p_value))
    I_percentage_list = np.array([3,70])
    mu = 0.001
    beta_list = np.linspace(0,1.5,41)*mu
    beta2_list = np.array([0,1,3,5])*mu
    times = 10
    args = []
    for filename in filename_list:
        args.append([filename,beta_list,beta2_list,mu,I_percentage_list,times])
    pool = Pool(processes=1)
    pool.map(run_one_network,args)