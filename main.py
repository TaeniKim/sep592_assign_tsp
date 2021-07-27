# -*- coding: utf-8 -*-
"""
Created on Sun Jul 18 16:24:29 2021

@author: taeni
"""
import time
import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.cluster import KMeans


# 2. Clustering - KMeans
def plot_KMeans(k_output):
    # 그래프 출력 (K-Means)
    fig = plt.figure(figsize= (15, 6))
    subplot = plt.subplot(1, 1, 1)
    subplot.set_title('Outputs of K-Means (k=5)', fontsize=14)
    subplot.scatter(k_output[:, 0], 
                    k_output[:, 1], 
                    c=k_output[:, 2], 
                    cmap=plt.cm.get_cmap('Accent', len(np.unique(k_output[:, 2]))),
                    marker='+',
                    s=20,
                    alpha=0.5)
    plt.show()

# Hyper parameter
def get_group_by_KMeans(nodes, is_plot=False):
    NUM_CLUSTER = 1  
    
    lst=[]
    for n, x, y in nodes:
        lst.append([x, y])
        
    k_cls = KMeans(n_clusters=NUM_CLUSTER).fit(lst)
    print('unique clusters of K-Means= ')
    print(np.unique(k_cls.labels_))

    # 그래프 출력할 데이터에 cluster labels 추가
    k_output = np.append(nodes, 
                        k_cls.labels_.reshape(-1, 1), 
                        axis=1)
    
    if is_plot:
        plot_KMeans(k_output)
    
    return k_output


# load tsp file
def load_file2df(filename, is_plot=False):
    with open(filename, 'r') as f:
        lines = f.readlines()
        lines = list(map(lambda x:x.strip(), lines))

        s = lines.index("NODE_COORD_SECTION")
        e = lines.index("EOF")
        parsed = lines[s+1: e]
        parsed = list(map(lambda x: tuple(map(lambda y: float(y), x.split())), parsed))
    
    # find_group
    lst = get_group_by_KMeans(parsed, is_plot)
    
    return pd.DataFrame(lst, columns=['node','x','y','group']).astype('int')


# nodes=[x, y, group]
def calc_distance(s, d):
    return math.sqrt((s['x'] - d['x'])**2 + (s['y'] - d['y'])**2)


def total_distance(paths, nodes):
    total = 0    
    for idx in range(len(paths)-1):        
        s = {'x':nodes.loc[paths[idx]]['x'], 'y':nodes.loc[paths[idx]]['y']}
        d = {'x':nodes.loc[paths[idx+1]]['x'], 'y':nodes.loc[paths[idx+1]]['y']}
        total += calc_distance(s, d)
    
    return total


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
                dist = calc_distance(s, t)
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


def exec_greedy(df, nodes):
    lst_node = df['node'].to_list()
    lst_x = nodes['x'].to_list()
    lst_y = nodes['y'].to_list()

    n = random.choice(lst_node)

    path = [n]
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
        #print(mIdx)
        
    return path

def search_greedy2(start_node, nodes):
    groups = nodes['group'].unique()
    path = []
    
    for group in groups:        
        df = nodes[nodes['group'] == group]
        print(group, len(df))
        start = time.time()        
        path.extend(exec_greedy(df, nodes))
        print('Elapse: ', time.time() - start)
        print('Distance: ', total_distance(path, nodes))        
        
    return path


if __name__ == '__main__':
    # 1. Load .tsp file
    filename = r'C:\Users\taeni\Documents\Travelling-Salesman-Problem\tsp/rl11849.tsp'
    nodes = load_file2df(filename, False)
    nodes.index = np.arange(1,len(nodes)+1)

    for i in range(10):
        start_node = 1
        paths = search_greedy2(start_node, nodes)
        dist = total_distance(paths, nodes)        
        print(f"{i}th total node and distance: {len(nodes)} --> {dist}")
        
        file = open(f'pathes_{i}.txt', 'w')
        for l in paths:
            file.write(str(l)+'\n')
        file.close()
    
