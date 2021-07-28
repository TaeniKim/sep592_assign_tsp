# -*- coding: utf-8 -*-
"""
Created on Sun Jul 18 16:24:29 2021

@author: taeni
"""
import os
import sys
import time
from datetime import datetime
import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.cluster import KMeans, DBSCAN



# 2. Clustering - KMeans
def plot_KMeans(output):
    # 그래프 출력 (K-Means)
    fig = plt.figure(figsize= (15, 6))
    subplot = plt.subplot(1, 1, 1)
    subplot.set_title('Outputs of K-Means (k=5)', fontsize=14)
    subplot.scatter(output[:, 1], 
                    output[:, 2], 
                    c=output[:, 3], 
                    cmap=plt.cm.get_cmap('Accent', len(np.unique(output[:, 3]))),
                    marker='+',
                    s=20,
                    alpha=0.5)

    fp = os.getcwd() + r'/result/'    
    dt = datetime.now().strftime('%Y-%m-%d %H%M%S')        
    plt.savefig(fp + dt + '_KMean.png')
    plt.close()

    

# Hyper parameter
def get_group_by_KMeans(nodes, is_plot=False):
    NUM_CLUSTER = 10
    
    lst=[]
    for n, x, y in nodes:
        lst.append([x, y])
        
    k_cls = KMeans(n_clusters=NUM_CLUSTER).fit(lst)
    print('unique clusters of K-Means= ')
    print(np.unique(k_cls.labels_))

    # 그래프 출력할 데이터에 cluster labels 추가
    output = np.append(nodes, 
                        k_cls.labels_.reshape(-1, 1), 
                        axis=1)
    
    if is_plot:
        plot_KMeans(output)
    
    return output


# Hyper parameter
def get_group_by_DBSCAN(nodes, is_plot=False):
    ## 파라미터 조정
    ## 클러스터 할당에 실패하면, "-1" 레이블이 발생함
    EPSILON = 280
    MIN_POINTS = 5
    
    lst=[]
    for n, x, y in nodes:
        lst.append([x, y])
    
    # DBSCAN 알고리즘 적용
    clusters = DBSCAN(eps=EPSILON, min_samples=MIN_POINTS).fit(lst)
    print('clusters= ')
    print(clusters.labels_)
    unique_labels = np.unique(clusters.labels_)
    print('unique clusters= ')
    print(unique_labels)
    
    # 그래프 출력할 데이터에 cluster labels 추가
    output = np.append(lst, 
                       clusters.labels_.reshape(-1, 1), 
                       axis=1)
    print('output= ')
    print(output)
    print()

    # 그래프 출력
    plt.figure(figsize= (24, 12))
    plt.scatter(output[:, 0], 
                output[:, 1], 
                c=output[:, 2], #color map
                cmap=plt.cm.get_cmap('Accent', len(unique_labels)),
                marker='+',
                s=20,
                alpha=0.5)
    cb = plt.colorbar()
    cb.set_ticks(range(len(unique_labels)))
    cb.set_ticklabels(['Cluster {:d}'.format(x) for x in range(len(unique_labels))] )
    
    return unique_labels


# load tsp file
def load_file2df(filename, is_plot=False):
    with open(filename, 'r') as f:
        lines = f.readlines()
        lines = list(map(lambda x:x.strip(), lines))

        s = lines.index("NODE_COORD_SECTION")
        e = lines.index("EOF")
        parsed = lines[s+1: e]
        parsed = list(map(lambda x: tuple(map(lambda y: int(float(y)), x.split())), parsed))
    
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
        print('Distance: ', total_distance(paths, nodes))        
        
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
        print('Distance: ', total_distance(paths, nodes))        
        
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
    print('Distance: ', total_distance(paths, nodes))        
        
    return paths, positions


def save_result(paths, pos, dist):
    fp = os.getcwd() + r'/result/'
    dt = datetime.now().strftime('%Y-%m-%d %H%M%S')        
    file = open(fp + dt + f'_pathes_{int(dist)}.txt', 'w')
    for l in paths:
        file.write(str(l)+'\n')
    file.close()
    
    # plotting
    lst_x = []
    lst_y = []
    for x, y in pos:
        lst_x.append(x)
        lst_y.append(y)        
    
    plt.figure(figsize=(24,12))
    plt.plot(lst_x, lst_y)   # plot A
    plt.title("Total Distance: {:,}".format(int(dist)))
    plt.text(lst_x[0], lst_y[0], "S", size=20, backgroundcolor='y')
    plt.text(lst_x[-1], lst_y[-1], "E", size=20, backgroundcolor='y')
    #plt.text(lst_x[0], lst_y[0], "S", size=20)
    #plt.text(lst_x[-1], lst_y[-1], "E", size=20)
    plt.savefig(fp + dt + f'_plot_{int(dist)}.png', bbox_inches='tight', pad_inches=0)
    plt.close()
    
def plot_by_paths(fp, nodes):
    with open(fp) as f:
        paths = f.readlines()
    paths = [int(x.strip()) for x in paths]
    
    
    # plotting
    lst_x = []
    lst_y = []
    for p in paths:
        lst_x.append(nodes.loc[p]['x'])
        lst_y.append(nodes.loc[p]['y'])
    
    plt.figure(figsize=(24,12))
    plt.plot(lst_x, lst_y)   # plot A
    plt.title("Total Distance: " + fp.split('/')[-2])
    plt.text(lst_x[0], lst_y[0], "S", size=20, backgroundcolor='y')
    plt.text(lst_x[-1], lst_y[-1], "E", size=20, backgroundcolor='y')
    plt.savefig(fp[:-3] + 'png', bbox_inches='tight', pad_inches=0)
    plt.close()   

'''
    fp = os.getcwd() + r'/result/' + 'solution.csv'
    plot_by_paths(fp, nodes)
'''


if __name__ == '__main__':
    # 1. Load .tsp file
    fp = os.getcwd() + r'/data/'    
    filename = fp + r'rl11849.tsp'
    nodes = load_file2df(filename, True)
    nodes.index = np.arange(1,len(nodes)+1)
    args = sys.argv
    for a in args:
        print(a)

    for i in range(1):
        start_node = 1
        
        #paths, pos = exec_group('greedy', nodes, args)
        #paths, pos = exec_group_fix('greedy', nodes, args)
        paths, pos = exec_all('greedy', nodes, args)
        
        dist = total_distance(paths, nodes)        
        print(f"{i}th total node and distance: {len(nodes)} --> {dist}")
        
        save_result(paths, pos, dist)     
    
