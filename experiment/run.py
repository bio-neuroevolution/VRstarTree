import time
from random import random

import numpy as np


from blocks import BTransaction, BlockChain
from dataloader import PocemonLoader
from utils import Configuration
from wise import simulation
from wise.blockDAG import analysis_utils
from wise.blockDAG import search_on_blockDAG as sob


def run_query_transaction():
    # 配置
    context = Configuration(max_children_num=16, max_entries_num=8, account_length=8, account_count=200,
                            select_nodes_func='', merge_nodes_func='', split_node_func='')

    # 读取数据
    print("VRTree读取数据...")
    dataLoader = PocemonLoader()
    df = dataLoader.refresh()                         #读取数据
    df = dataLoader.extend_df(df=df,repeat_times=1)   #扩大数据
    df = dataLoader.normalize(df)                     #归一化
    transactions = [BTransaction(row['type'],row['lon'],row['lat'],row['ts'],None) for index,row in df.iterrows()]

    #创建和分配账户
    account_name,account_dis = dataLoader.create_account_name(context.account_count,context.account_length)
    accs = np.random.choice(account_name,len(transactions))
    diss = [account_dis[account_name.index(a)] for a in accs]
    for i,tr in enumerate(transactions):
        tr.account = accs[i]


    blocksizes = [30,40,50,60,70,80,90,100,110]
    for i,blocksize in enumerate(blocksizes):
        print("VRTree创建区块,blocksize="+blocksize+"...")
        # 创建区块链
        chain  = BlockChain(context,blocksize,transactions=transactions)
        # 创建查询分布
        mbrs = chain.create_query(count=100,sizes=[2,2,2],posrandom=100,lengthcenter=0.05,lengthscale=0.025)

        # 执行查询
        print("VRTree执行查询...")
        begin = time.time()
        for mbr in mbrs:
            chain.query_tran(mbr)
        print("VRTree交易查询消耗（优化前）:" + str(time.time() - begin))

        # 根据查询优化
        chain.optima()

        # 第二次交易查询
        begin = time.time()
        for mbr in mbrs:
            chain.query_tran(mbr)
        print("VRTree交易查询消耗（优化后）:" + str(time.time() - begin))

        # 创建block_dag区块(用于对比，来自https://github.com/ILDAR9/spatiotemporal blockdag.)
        print('创建BlockDAG...')
        settings = dict(repeat_times=1, tr=60, D=3, bs=blocksize, alpha=10)
        block_dag = simulation.GeneratorDAGchain.generate(**settings)

        # 创建查询
        query_ranges = []
        for i,mbr in enumerate(mbrs):
            bs1 = mbr.boundary(0)  #经度范围
            bs2 = mbr.boundary(1)  #纬度范围
            minp = (bs1[0]*(block_dag.max_lat-block_dag.min_lat)+block_dag.min_lat,
                    bs2[0]*(block_dag.max_lon-block_dag.min_lon)+block_dag.min_lon)
            maxp = (bs1[1]*(block_dag.max_lat-block_dag.min_lat)+block_dag.min_lat,
                    bs2[1]*(block_dag.max_lon-block_dag.min_lon)+block_dag.min_lon)
            t_start,t_end = block_dag.min_ts,block_dag.max_ts
            query_ranges.append({'minp':minp,'maxp':maxp,'t_start':t_start,'t_end':t_end})

        # 执行查询
        print("BlockDAG执行查询...")
        begin = time.time()
        for i,query_range in enumerate(query_ranges):
            min_point, max_point, t_start, t_end = query_range['minp'],query_range['maxp'],query_range['t_start'],query_range['t_end']
            min_point, max_point = analysis_utils.__to_Cartesian_rect(min_point, max_point)
            rng = analysis_utils.__measure_time(sob.kd_range, 1, block_dag, min_point, max_point, t_start, t_end)

        print("BlockDAG交易查询消耗:" + str(time.time() - begin))

if __name__ == '__main__':
    run_query_transaction()