# -*- coding: utf-8 -*-
"""
Created on Sun Jul 18 16:24:29 2021

@author: taeni
"""
import os
import sys
import time
import argparse
from datetime import datetime
import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

import common as cm
import greedy
import ga

if __name__ == '__main__':
    # args
    parser = argparse.ArgumentParser(description='TSP solver args.')
    parser.add_argument('file_name', help='tsp filename')
    parser.add_argument('--algorism', '-a', type=str, default='greedy',
                        help='algorism: greedy or ga')
    parser.add_argument('--p_size', '-p', type=int, default=10,
                        help='population size')
    parser.add_argument('--g_limit', '-gl', type=int, default=10,
                        help='ga: generation limit')
    parser.add_argument('--g_fit', '-gfit', type=str, default='greedy',
                        help='ga: fitness - greedy, random')
    parser.add_argument('--s_elite_rate', '-ser', type=float, default=0.2,
                        help='ga: selection elite rate')
    parser.add_argument('--s_tour_rate', '-str', type=float, default=0.4,
                        help='ga: selection tournement rate')
    parser.add_argument('--c_size_min', '-csmin', type=int, default=10,
                        help='ga: crossover size min')
    parser.add_argument('--c_size_max', '-csmax', type=int, default=50,
                        help='ga: crossover size max')
    parser.add_argument('--m_rate', '-mr', type=float, default=0.2,
                        help='ga: mutation rate')
    parser.add_argument('--m_size_min', '-msmin', type=int, default=3,
                        help='ga: mutation size min')
    parser.add_argument('--m_size_max', '-msmax', type=int, default=10,
                        help='ga: mutation size max')    
    parser.add_argument('--h_ratio', '-hr', type=float, default=0.5,
                        help='heuristic ratio')
    parser.add_argument('--gd_method', '-gm', type=int, default=1,
                        help='Greedy method selection, 0-greedy, 1-greedy+heuristic')    
    parser.add_argument('--gd_cluster_use', '-gc', type=int, default=0,
                        help='Greedy method selection, 0-not, 1-use')
    args = parser.parse_args()    
    
    fp_data = os.getcwd() + '/data/'
    fp = fp_data + args.file_name
    if not os.path.exists(fp):
        print(f'file ({args.file_name}) is not exist..!!')
        sys.exit()
    
    if args.algorism == 'ga':        
        solution, s_dist, nodes = ga.ga(args)
    else:        
        solution, pos = greedy.greedy(args)
        