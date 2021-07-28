# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 21:52:46 2021

@author: taeni
"""
import os
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
import math

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
