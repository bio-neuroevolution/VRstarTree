

import  chart_studio
#from chart-studio.plotly.offline import init_notebook_mode, iplot
#import plotly.plotly as py
#init_notebook_mode(connected=True)
from importlib import reload
from datetime import datetime
import time

import sys
sys.path.append('../wise')
import utils
from blockDAG import blockDAG
from merkleKDtree import merkle_kdtree
from blockDAG import search_on_blockDAG as sob
from blockDAG import geo_utils as geo
import simulation
from blockDAG import analysis_utils as au

#reload(au)
#reload(merkle_kdtree)
#reload(blockDAG)
#reload(simulation)

settings = dict(repeat_times = 1, tr=60, D=3, bs=40, alpha=10)
block_dag = simulation.GeneratorDAGchain.generate(**settings)

# Range
#reload(merkle_kdtree)
#reload(sob)
#reload(au)

mn = (22.20, 113.89)
mx = (22.31, 113.95)
t_start = au.get_ts(2017, 1, 2, 0, 0, 0)
t_end = au.get_ts(2019, 1, 3, 0, 0, 0)

au.compare_time_range(block_dag, mn, mx, t_start, t_end, range_count=10)

block_dag.super_optimize()

# k-NN
reload(sob)
reload(au)

q_point = (114, 23)
count_nn = 20
t_start = au.get_ts(2017, 2, 2, 0, 0, 0)
t_end = au.get_ts(2019, 2, 3, 0, 0, 0)
range_count = 8

au.compare_time_knn(block_dag, q_point, count_nn, t_start, t_end, range_count)

# k-NN bound
reload(sob)
reload(au)

q_point = (22.6, 114)
count_nn = 20
t_start = au.get_ts(2017, 2, 2, 0, 0, 0)
t_end = au.get_ts(2019, 2, 3, 0, 0, 0)
bound = 12
range_count = 8

au.compare_time_knn_bound(block_dag, q_point, count_nn, bound, t_start, t_end, range_count)

# query Ball-point
reload(sob)
reload(au)

q_point = (22.6, 114)
t_start = au.get_ts(2017, 2, 2, 0, 0, 0)
t_end = au.get_ts(2019, 2, 3, 0, 0, 0)
r = 11
range_count = 8

au.compare_time_query_ball(block_dag, q_point, r, t_start, t_end, range_count)


#2  Experiments
#2.1  Range
reload(merkle_kdtree)
reload(blockDAG)
reload(sob)
reload(simulation)
reload(au)

algo = 'range'
df = au.experiment_bs_and_spatiotemporal(range_count=10, algo=algo)

algo = 'range'
df = au.experiment_range_growing_blockchain(range_count=5, algo=algo)

# 2.2  k-NN
reload(merkle_kdtree)
reload(blockDAG)
reload(sob)
reload(simulation)
reload(au)

algo = 'knn'
df = au.experiment_bs_and_spatiotemporal(range_count=10, algo=algo)

df = au.experiment_range_growing_blockchain(range_count=5, algo=algo)

#2.3  k-NN bound
reload(merkle_kdtree)
reload(blockDAG)
reload(sob)
reload(simulation)
reload(au)

algo = 'knn_bound'
df = au.experiment_bs_and_spatiotemporal(range_count=10, algo=algo)

df = au.experiment_range_growing_blockchain(range_count=5, algo=algo)

# 2.4  Ball point
reload(merkle_kdtree)
reload(blockDAG)
reload(sob)
reload(simulation)
reload(au)

algo = 'query_ball'
df = au.experiment_bs_and_spatiotemporal(range_count=10, algo=algo)

df = au.experiment_range_growing_blockchain(range_count=5, algo=algo)

# 2.5  Vary k for k-NN

reload(merkle_kdtree)
reload(blockDAG)
reload(sob)
reload(simulation)
reload(au)

df = au.experiment_knn_vary_k(range_count=11)

# 2.6  Vary boud for k-NN bound
reload(merkle_kdtree)
reload(blockDAG)
reload(sob)
reload(simulation)
reload(au)

df = au.experiment_knn_bound_vary_b(range_count=11)

# 2.7  Vary r for ball-poit

reload(merkle_kdtree)
reload(blockDAG)
reload(sob)
reload(simulation)
reload(au)

df = au.experiment_ball_point_vary_r(range_count=11)

# 3  Check new queries
df = simulation.GeneratorDAGchain.read_prepare_dataset()
datetime.utcfromtimestamp(block_dag.chain[-1]['end_time'])
reload(geo)
geo.haversine(40,40,41,40)
geo.haversine(40,40,40,41)
geo.haversine(40,40,41,41)