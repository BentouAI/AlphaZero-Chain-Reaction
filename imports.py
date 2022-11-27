# imports
import torch
import numpy as np
import pygame
# importing dotted dictionary to access dictionary members through dot
from dotted_dict import DottedDict
from torch.utils.data import Dataset, DataLoader
import os
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
from scipy.interpolate import make_interp_spline, BSpline
from joblib import Parallel, delayed
import copy
import random
import time

args = DottedDict({
    'M':5,
    'N':5,
    'device':torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'cpuct':2,
    'epsilon':0.25,
    'dropout':0.3,
    'batch_size':128,
    'playout_cap_p':0.35,
    'mcts_iter':1000,
    'mcts_iter_fast':200,
    'stopped_and_now_start':True
})

'''
Dirichlet noise in the alphazero paper was added to the prior 
probabilities in the root node; this was scaled in inverse proportion 
to the approximate number of legal moves in a typical position
here the max number of legal moves would be args.M*args.N
and the min number of legal moves would be 1
On average, the no. of legal moves is more likely to be more than
average for chain reaction, so we choose (args.M*args.N) * 0.75 here
for chess, it's around 35 and dirichlet alpha was chosen to be
0.3, which gives the proportionality constant to be 10.5
alpha = 0.03 for Go with 270 legal moves gives the constant to be 8
we choose it 10 here, 
Ref: https://stats.stackexchange.com/questions/322831/purpose-of-dirichlet-noise-in-the-alphazero-paper
'''

args.FOLDER = str(args.M*args.N)

args.dirichlet_alpha = 10/((args.M*args.N)*0.75)
if args.playout_cap_p!=1:
    args.do_playout_cap = True
    # args.mcts_iter_full = int(args.mcts_iter_fast + (args.mcts_iter - args.mcts_iter_fast)//args.playout_cap_p)
    args.mcts_iter_full = args.mcts_iter
else:
    args.do_playout_cap = False

# Ref: https://medium.com/oracledevs/lessons-from-alpha-zero-part-6-hyperparameter-tuning-b1cfcbe4ca9a

args.hist_start = 4
args.hist_end = 50
args.max_orbs = (args.M-2)*(args.N-2)*3 + 2*(args.M-2 + args.N-2)*2 + 4

# assuming on average, 0.8 times the max number of moves are played each game
args.moves_per_game = int(0.8 * args.max_orbs)
args.threads = 8

# data augmentation adds a multiple of 8
args.buffer_start = int(args.moves_per_game * args.threads * 8 * args.hist_start * args.playout_cap_p)
args.buffer_end = int(args.moves_per_game * args.threads * 8 * args.hist_end * args.playout_cap_p)
args.slow_window_generations = 35

if args.stopped_and_now_start:
    args.buffer_start = args.buffer_end

# an upper bound on the number of states
args.complexity = int((args.M-2)*(args.N-2) * np.log10(7) + 2*(args.M-2 + args.N-2) * np.log10(5) + 4*np.log10(3))
