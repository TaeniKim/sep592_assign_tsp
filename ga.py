# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 12:42:32 2021

@author: taeni
"""
import os
import time
from datetime import datetime
import random
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import common as cm

# load tsp file
def load_tspfile(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        lines = list(map(lambda x:x.strip(), lines))

        s = lines.index("NODE_COORD_SECTION")
        e = lines.index("EOF")
        parsed = lines[s+1: e]
        parsed = list(map(lambda x: tuple(map(lambda y: int(float(y)), x.split())), parsed))
    
    
    return pd.DataFrame(parsed, columns=['node','x','y']).astype('int')


# 전체 수행
def exec_greedy4init(nodes, ratio=0.5, s_node=5786, e_node=7340):
    start = time.time()
    print(f'Start greedy4init -- s:{s_node}, e:{e_node}')
    lst_node = nodes['node'].to_list()
    lst_x = nodes['x'].to_list()
    lst_y = nodes['y'].to_list()

    lst_h = []
    for i in lst_node:
        dist_h = math.sqrt((lst_x[e_node-1] - lst_x[i-1])**2 +\
                             (lst_y[e_node-1] - lst_y[i-1])**2)
        lst_h.append(dist_h)

    path = [s_node]
    pos = []
    pos.append([lst_x[s_node-1], lst_y[s_node-1]])
    toVisit = lst_node
    toVisit.remove(s_node)
    while len(toVisit) > 0:
        m = 999999999
        mIdx = -1
        for target in toVisit:
            #print(target, path[-1])
            dist = math.sqrt((lst_x[target-1] - lst_x[path[-1]-1])**2 +\
                             (lst_y[target-1] - lst_y[path[-1]-1])**2)
            dist -= (lst_h[target-1]) * ratio
            if dist < m:
                m = dist
                mIdx = target
        
        toVisit.remove(mIdx)
        path.append(mIdx)
        pos.append([lst_x[mIdx-1], lst_y[mIdx-1]])
        
    print('Elapse Time: ', time.time() - start, '\n')
    return path, pos

def calc_population_dist(population, nodes):
    print('calc_population_dist Start')
    dist = {}
    for c in population.columns:
        paths = population[c].tolist()
        dist[c] = cm.total_distance(paths, nodes)
        print(f'{c} -- {dist[c]:.2f}')
        
    return dist  

# fitness - minimum total distance 
def fitness(population, nodes):
    return calc_population_dist(population, nodes)


# Selection = Elitism + Tournaments + Random (Default: 2:4:4)
def selection(population, nodes):
    # 2. Evaluate Fitness
    print('\nevaluation')
    dist = calc_population_dist(population, nodes)    
    dist = {k: v for k, v in sorted(dist.items(), key=lambda item: item[1])}
    cols = list(dist.keys())
    
    n_population = pd.DataFrame()    
    # Elitism
    elitism_cols = cols[:SELECTION_SIZE_ELITE]
    rest_cols = cols[SELECTION_SIZE_ELITE:]
    for i in range(SELECTION_SIZE_ELITE):
        n_population[i] = population[elitism_cols[i]]
    
    # Tournaments    
    for i in range(SELECTION_SIZE_TOURNAMENT):
        buf = random.sample(rest_cols, 2)
        if buf[0] > buf[1]:
            n_population[i+SELECTION_SIZE_ELITE] = population[buf[0]]
        else:
            n_population[i+SELECTION_SIZE_ELITE] = population[buf[1]]

    # Random
    paths = nodes['node'].to_list()
    for i in range(SELECTION_SIZE_RANDOM):
        # use greedy & heuristic
        buf = random.sample(paths,2)
        s_node = buf[0]
        e_node = buf[1]
           
        path, pos = exec_greedy4init(nodes, 0.5, s_node, e_node)
        n_population[i+SELECTION_SIZE_ELITE+SELECTION_SIZE_TOURNAMENT] = path

    dist = calc_population_dist(n_population, nodes)
    dist = {k: v for k, v in sorted(dist.items(), key=lambda item: item[1])}
    cols = list(dist.keys())
    
    solution = n_population[cols[0]].to_list()

    return n_population, dist, solution

# Partially - Mapped Crossover
# random crossover size(default 10~50)
# p1, p2 crossover index random
def crossover(population, nodes):
    cols = population.columns.to_list()
    for idx in range(0, int(POPULATION_SIZE), 2):
        p1 = population[cols[idx]].tolist()
        p2 = population[cols[idx+1]].tolist()
        
        p3 = p1[:]
        p4 = p2[:]
    
        size = random.randrange(CROSSOVER_SIZE_MIN, CROSSOVER_SIZE_MAX+1)
        
        p1_idx = random.choice(range(0, len(nodes)-size))
        p2_idx = random.choice(range(0, len(nodes)-size))

        print(p1_idx, p2_idx)
        # find & replace & crossover
        for i in range(size):            
            p1_cross = p3[p1_idx:(p1_idx + size)]
            p2_cross = p4[p2_idx:(p2_idx + size)]            
            print('crossover: ', cols[idx], size, p1_cross[i], ' <-> ', p2_cross[i])        
            p3[p3.index(p2_cross[i])] = p1_cross[i]
            p3[p1_idx+i] = p2_cross[i]
            p4[p4.index(p1_cross[i])] = p2_cross[i]
            p4[p2_idx+i] = p1_cross[i]
            
        population[cols[idx]] = p3
        population[cols[idx+1]] = p4
        
    return population

# 
def mutate(population, nodes):
    cols = population.columns.to_list()
    for i in range(POPULATION_SIZE):
        p = random.random()
        if p < MUTATION_RATE:            
            size = random.randrange(MUTATION_SIZE_MIN, MUTATION_SIZE_MAX+1)            
            idx1 = random.choice(range(0, len(nodes)-size))
            idx2 = random.choice(range(0, len(nodes)-size))
            
            s = set(range(idx1, idx1+size))
            while len(s.intersection(range(idx2, idx2+size))) > 0:
                  idx2 = random.choice(range(0, len(nodes)-size))

            print('Mutate: ', cols[i], size, idx1, ' <-> ', idx2)
            buf = population[cols[i]].tolist()
            p1_cross = buf[idx1:(idx1+size)]
            p2_cross = buf[idx2:(idx2+size)]
            
            # find & replace & crossover
            for j in range(size):
                buf[idx1+j] = p2_cross[j]
                buf[idx2+j] = p1_cross[j]
            
            population[cols[i]] = buf            
        
    return population


def init_solution(f_sol, nodes):
    # check solution file
    solution = None
    dist = 99999999
    if os.path.exists(f_sol):
        print('init_solution -- Read solution file')
        with open(f_sol) as f:
            solution = f.readlines()            
        solution = [int(x.strip()) for x in solution]
        dist = cm.total_distance(solution, nodes)
        print('init_solution -- dist: ', dist)

    return solution, dist


def init_population(f_pop, nodes):
    population = pd.DataFrame()
    
    # check population file
    pop_file_exist = False
    if os.path.exists(f_pop):
        pop_file_exist = True
        
    if pop_file_exist:
        print('init_population -- Read population file')
        population = pd.read_csv(f_pop)
    else:
        paths = nodes['node'].to_list()
        for i in range(POPULATION_SIZE):
            if False:
                # create random populations
                buf = paths[:]
                random.shuffle(buf)
                population[i] = buf
            else:
                # use greedy & heuristic
                buf = random.sample(paths,2)
                s_node = buf[0]
                e_node = buf[1]
                   
                path, pos = exec_greedy4init(nodes, 0.5, s_node, e_node)
                population[i] = path
                
        save_population(f_pop, population)
    
    return population


def save_result(solution, dist, nodes):
    fp = os.getcwd() + r'/result/'
    dt = datetime.now().strftime('%Y-%m-%d %H%M%S')        
    file = open(fp + dt + f'_pathes_{int(dist)}.txt', 'w')
    for l in solution:
        file.write(str(l)+'\n')
    file.close()
    
    # plotting
    lst_x = []
    lst_y = []
    for i in range(len(nodes)):
        lst_x.append(nodes.loc[solution[i]]['x'])
        lst_y.append(nodes.loc[solution[i]]['y'])        
    
    plt.figure(figsize=(24,12))
    plt.plot(lst_x, lst_y)   # plot A
    plt.title("Total Distance: {:,}".format(int(dist)))
    plt.text(lst_x[0], lst_y[0], "S", size=20, backgroundcolor='y')
    plt.text(lst_x[-1], lst_y[-1], "E", size=20, backgroundcolor='y')
    plt.savefig(fp + dt + f'_plot_{int(dist)}.png', bbox_inches='tight', pad_inches=0)
    plt.close()
    

def save_solution(f_sol, solution):
    print('save_solution..!!')
    file = open(f_sol, 'w')
    for i in range(len(solution)):
        file.write(str(solution[i])+'\n')
    file.close()
    
    # history
    fp = os.getcwd() + r'/generation/'
    dt = datetime.now().strftime('%Y-%m-%d %H%M%S')        
    file = open(fp + dt + '_solution.csv', 'w')
    for l in solution:
        file.write(str(l)+'\n')
    file.close()


def save_population(f_pop, population):
    # check solution file
    population.to_csv(f_pop, index=False)
    print('save_population..!!')
    
def update_population_dist(population, dist):
    # check solution file
    population.loc['dist'] = dist
    
def update_population_column(population, col, paths):
    # check solution file
    # dist 행이 없어야 한다.. 행의 길이가 같아야함
    # population = population.drop('dist')
    population[str(col)] = paths    


def init_args(args):    
    global POPULATION_SIZE
    global GENERATION_LIMIT
    global SELECTION_SIZE_ELITE
    global SELECTION_SIZE_TOURNAMENT
    global SELECTION_SIZE_RANDOM
    global CROSSOVER_SIZE_MIN
    global CROSSOVER_SIZE_MAX
    global MUTATION_RATE
    global MUTATION_SIZE_MIN
    global MUTATION_SIZE_MAX
    
    GENERATION_LIMIT = args.g_limit
    POPULATION_SIZE = args.p_size 
    SELECTION_SIZE_ELITE = int(POPULATION_SIZE * args.s_elite_rate)
    SELECTION_SIZE_TOURNAMENT = int(POPULATION_SIZE * args.s_tour_rate)
    SELECTION_SIZE_RANDOM = POPULATION_SIZE - (SELECTION_SIZE_ELITE + SELECTION_SIZE_TOURNAMENT)
    CROSSOVER_SIZE_MIN = args.c_size_min
    CROSSOVER_SIZE_MAX = args.c_size_max
    MUTATION_RATE = args.m_rate
    MUTATION_SIZE_MIN = args.m_size_min
    MUTATION_SIZE_MAX = args.m_size_max


## solution = (fitness, [])
def ga(args):
    # 0. Standby - load tsp file, check option...
    # 1. Initial Population
    # 2. Evaluate Fitness & Select
    # 3. Crossover
    # 4. Mutation
    # 5. Next-Generation
    # Loop 2~5

    init_args(args)
    
    # 0. Load .tsp file
    fp = os.getcwd() + r'/data/'
    filename = fp + args.file_name
    nodes = load_tspfile(filename)
    nodes.index = np.arange(1,len(nodes)+1)
    

    
    # 1. Initial Population
    f_sol = os.getcwd() + r'/generation/solution.csv'
    solution, s_dist = init_solution(f_sol, nodes)
    f_pop = os.getcwd() + r'/generation/population.txt'
    population = init_population(f_pop, nodes)
        
    count = 0
    while count < GENERATION_LIMIT:        
        count += 1
        print(f'\ngen = {count}')
        
        print('\nselection')
        n_population, dist, sol = selection(population, nodes)
        if dist[list(dist.keys())[0]] < s_dist:
            s_dist = dist[list(dist.keys())[0]]
            solution = sol
            save_solution(f_sol, solution)
            save_population(f_pop, population)
        
        print('\ncrossover')
        population = crossover(population, nodes)
        
        print('\nmutate')
        population = mutate(population, nodes)\
            
    print(f"total distance: {s_dist}")    
    save_result(solution, s_dist, nodes)

    return solution, s_dist, nodes

if __name__ == '__main__':
    solution, s_dist, nodes = ga()
    
    