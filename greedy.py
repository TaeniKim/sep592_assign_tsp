# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 21:51:55 2021

@author: taeni
"""
import os
import time
from datetime import datetime
import random
import math
import numpy as np

import common as cm


def check_skip_nodes(df, lst_skip, s, t):
    diff_x = abs(s['x'] - t['x'])
    diff_y = abs(s['y'] - t['y'])
        
    # (+,+)
    df_buf = df[(df['x']>(s['x'] + diff_x)) & (df['y']>(s['y'] + diff_y))]
    for idx in range(len(df_buf)):
        node = df_buf.iloc[idx]['node']
        lst_skip[node-1] = True
    #print('Skip (+,+) -->', len(df_buf))
        
    # (+,-)
    df_buf = df[(df['x']>(s['x'] + diff_x)) & (df['y']<(s['y'] - diff_y))]
    for idx in range(len(df_buf)):
        node = df_buf.iloc[idx]['node']
        lst_skip[node-1] = True
    #print('Skip (+,-) -->', len(df_buf))
        
    # (-,+)
    df_buf = df[(df['x']<(s['x'] - diff_x)) & (df['y']>(s['y'] + diff_y))]
    for idx in range(len(df_buf)):
        node = df_buf.iloc[idx]['node']
        lst_skip[node-1] = True
    #print('Skip (-,+) -->', len(df_buf))
        
    # (-,-)
    df_buf = df[(df['x']<(s['x'] - diff_x)) & (df['y']<(s['y'] - diff_y))]
    for idx in range(len(df_buf)):
        node = df_buf.iloc[idx]['node']
        lst_skip[node-1] = True
    #print('Skip (-,-) -->', len(df_buf))

    return lst_skip


def search_greedy(start_node, nodes):
    groups = nodes['group'].unique()
    lst_visited=[]    
    for i in range(len(nodes)):
        lst_visited.append(False)
    
    for group in groups:        
        df = nodes[nodes['group'] == group]
        print(group, len(df))
        start_node = int(df.iloc[0]['node'])
        path = [start_node]
        lst_visited[start_node-1] = True
        
        cnt = 0        
        for idx in range(len(df)):
            s = {'x':df.loc[start_node]['x'], 'y':df.loc[start_node]['y']}
            #print('s: ', s)
            buf_min = 99999999

            lst_skip=[]                        
            for i in range(len(nodes)):
                lst_skip.append(False)

            start = time.time()
            for idx in range(len(df)):
                node = df.iloc[idx]['node']
                if lst_visited[node-1]:
                    continue                        

                if lst_skip[node-1]:
                    continue

                t = {'x':df.iloc[idx]['x'], 'y':df.iloc[idx]['y']}
                dist = cm.calc_distance(s, t)
                if dist < buf_min:
                    buf_min = dist
                    start_node = node
                    lst_skip = check_skip_nodes(df, lst_skip, s, t)
                    #print('t: ', t)
            
            path.append(start_node)
            lst_visited[start_node-1] = True
            cnt += 1
            print(start_node, f'\t\t {cnt} / {len(df)}')
            print('Elapse: ', time.time() - start)
        
    return path


def exec_randomEx(nodes):
    lst_node = nodes['node'].to_list()
    lst_x = nodes['x'].to_list()
    lst_y = nodes['y'].to_list()

    #s_node = 5786 # Left, Bottom
    s_node = 7340 # Right, Top

    path = [s_node]
    pos = []
    pos.append([lst_x[s_node-1], lst_y[s_node-1]])
    toVisit = lst_node
    toVisit.remove(s_node)
    while len(toVisit) > 0:
        mIdx = random.choice(toVisit)
        
        toVisit.remove(mIdx)
        path.append(mIdx)
        pos.append([lst_x[mIdx-1], lst_y[mIdx-1]])
        
    return path, pos


# 전체 수행
def exec_greedyEx(nodes, ratio=1.0):
    lst_node = nodes['node'].to_list()
    lst_x = nodes['x'].to_list()
    lst_y = nodes['y'].to_list()

    s_node = 5786 # Left, Bottom
    e_node = 7340 # Right, Top

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
        
    return path, pos


# group 단위 수행 용
def exec_greedyEx2(df, nodes, ratio=1.0):
    lst_node = df['node'].to_list()
    lst_x = nodes['x'].to_list()
    lst_y = nodes['y'].to_list()
    lst_nodes = nodes['node'].to_list()

    s_node = 5786 # Left, Bottom
    e_node = 7340 # Right, Top
    
    is_first = False
    for n in lst_node:
        if n == s_node:
            is_first = True 
        
    if not is_first:
        s_node = random.choice(lst_node)
        
    print('Start Node', s_node)
        
    lst_h = []
    for i in lst_nodes:
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
            # print(target)
            dist = math.sqrt((lst_x[target-1] - lst_x[path[-1]-1])**2 +\
                             (lst_y[target-1] - lst_y[path[-1]-1])**2)
            dist -= (lst_h[target-1]) * ratio
            if dist < m:
                m = dist
                mIdx = target
        
        toVisit.remove(mIdx)
        path.append(mIdx)
        pos.append([lst_x[mIdx-1], lst_y[mIdx-1]])
        
    return path, pos


# greedy 기본 - 그룹단위 수행
def exec_greedy(df, nodes):
    lst_node = df['node'].to_list()
    lst_x = nodes['x'].to_list()
    lst_y = nodes['y'].to_list()

    n = random.choice(lst_node)

    path = [n]
    pos = []
    pos.append([lst_x[n-1], lst_y[n-1]])
    toVisit = lst_node
    toVisit.remove(n)
    while len(toVisit) > 0:
        m = 999999999
        mIdx = -1
        for target in toVisit:
            #print(target, path[-1])
            dist = math.sqrt((lst_x[target-1] - lst_x[path[-1]-1])**2 +\
                             (lst_y[target-1] - lst_y[path[-1]-1])**2)
            if dist < m:
                m = dist
                mIdx = target
        
        toVisit.remove(mIdx)
        path.append(mIdx)
        pos.append([lst_x[mIdx-1], lst_y[mIdx-1]])
        #print(mIdx)
        
    return path, pos


# group 무작위 진행
def exec_group(algo, nodes, args):
    groups = nodes['group'].unique()
    paths = []
    positions = []
    
    for group in groups:        
        df = nodes[nodes['group'] == group]
        print(group, len(df))
        start = time.time()
        print('Start groups --', algo)
        path, pos = exec_greedy(df, nodes)
        paths.extend(path)
        positions.extend(pos)
        print('Elapse: ', time.time() - start)
        print('Distance: ', cm.total_distance(paths, nodes))        
        
    return paths, positions


# group 순서 선별 하여 수행
def exec_group_fix(algo, nodes, args):
    groups = nodes['group'].unique().tolist()
    lst_x = nodes['x'].to_list()
    lst_y = nodes['y'].to_list()
    
    paths = []
    positions = []

    # find group order
    first_group = nodes.loc[5786]['group']
    final_group = nodes.loc[7340]['group']    
    groups.remove(first_group)
    groups.remove(final_group)
    new_groups = []
    new_groups.append(first_group)
    
    lst_g = []
    for g in groups:
        df = nodes[nodes['group'] == g]
        df = df.sort_values(by=['x', 'y'])
        buf = df.iloc[int(len(df)/2)]
        dist = math.sqrt((lst_x[7340-1] - buf['x'])**2 +\
                         (lst_y[7340-1] - buf['y'])**2)
        lst_g.append((dist, buf['group']))
    
    lst_g = sorted(lst_g, key=lambda s:s[0], reverse=True)
    for d, g in lst_g:
        new_groups.append(g)
        
    new_groups.append(final_group)
    groups = new_groups    
    print('Groups: ', groups)    
    
    for group in groups:        
        df = nodes[nodes['group'] == group]
        print(group, len(df))
        start = time.time()
        print('Start groups --', algo)
        path, pos = exec_greedyEx2(df, nodes, ratio=0.25)
        paths.extend(path)
        positions.extend(pos)
        print('Elapse: ', time.time() - start)
        print('Distance: ', cm.total_distance(paths, nodes))        
        
    return paths, positions


def exec_all(algo, nodes, args):
    paths = []
    positions = []
    ratio = 0.1
    
    start = time.time()
    print('Start all --', algo, f'ratio: {ratio}')
    paths, positions = exec_greedyEx(nodes, ratio)
    #paths, positions = exec_randomEx(nodes)
    print('Elapse: ', time.time() - start)
    print('Distance: ', cm.total_distance(paths, nodes))        
        
    return paths, positions


def init_args(args):    
    global START_NODE
    global HEURISTIC_RATIO
    global METHOD_NO    
    
    START_NODE = args.g_limit
    HEURISTIC_RATIO = args.p_size 
    METHOD_NO = args.gd_method


def greedy(args):
    
    init_args(args)
    
    # 1. Load .tsp file
    fp = os.getcwd() + r'/data/'    
    filename = fp + args.file_name
    nodes = cm.load_file2df(filename, True)
    nodes.index = np.arange(1,len(nodes)+1)
    
    #paths, pos = exec_group('greedy', nodes, args)
    #paths, pos = exec_group_fix('greedy', nodes, args)
    paths, pos = exec_all('greedy', nodes, args)
    
    dist = cm.total_distance(paths, nodes)        
    print(f"total node and distance: {len(nodes)} --> {dist}")
    
    cm.save_result(paths, pos, dist)
    
    return paths, dist

    
    