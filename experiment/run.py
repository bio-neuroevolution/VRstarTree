import time
from random import random

import numpy as np
import matplotlib.pyplot as plt

from log import LogHandler



from blocks import BTransaction, BlockChain
from dataloader import PocemonLoader
from utils import Configuration
from wise import simulation
from wise.blockDAG import analysis_utils
from wise.blockDAG import search_on_blockDAG as sob

# 读取数据
logging = LogHandler('run')

def query_transaction(context,blocksizes,content='all',region_params=None):
    """
    一次查询实验
    :param conetxt 配置信息
    :param blocksizes 区块大小列表
    :param region_params 生成立方体实验数据参数
    """
    logging.info("VRTree读取数据...")
    dataLoader = PocemonLoader()
    df = dataLoader.refresh()                         #读取数据
    #df = dataLoader.extend_df(df=df,repeat_times=1)   #扩大数据
    df = dataLoader.normalize(df)                     #归一化
    transactions = [BTransaction(row['type'],row['lon'],row['lat'],row['ts'],None) for index,row in df.iterrows()]
    if region_params is not None:
        dataLoader.create_region(transactions,
                                 geotype_probs=region_params['geotype_probs'],
                                 length_probs=region_params['length_probs'],
                                 lengthcenters=region_params['lengthcenters'],
                                 lengthscales=region_params['lengthscales'])

    #创建和分配账户
    account_name,account_dis = dataLoader.create_account_name(context.account_count,context.account_length)
    accs = np.random.choice(account_name,len(transactions))
    diss = [account_dis[account_name.index(a)] for a in accs]
    for i,tr in enumerate(transactions):
        tr.account = accs[i]


    rtreep,rtreep_nodecount,rtreea,rtreea_nodecount,kdtree = [],[],[],[],[]
    for i,blocksize in enumerate(blocksizes):
        logging.info("blocksize=" + str(blocksize))
        # 创建查询分布
        mbrs = BlockChain.create_query(count=200, sizes=[2, 2, 2], posrandom=100, lengthcenter=0.05, lengthscale=0.1)

        if content == 'all' or content.__contains__('rtree'):

            logging.info("VRTree创建区块...")
            # 创建区块链
            chain  = BlockChain(context,blocksize,transactions=transactions)
            logging.info("VRTree创建区块完成，交易VR*-tree节点数" + str(chain.tran_nodecount()))


            # 执行查询
            logging.info("VRTree执行查询...")
            begin = time.time()
            for mbr in mbrs:
                chain.query_tran(mbr)
            rtreep.append(time.time() - begin)
            rtreep_nodecount.append(chain.query_tran_node_count)
            logging.info("VRTree交易查询消耗（优化前）:" + str(rtreep[-1])+'，访问节点数：'+str(chain.query_tran_node_count))

        if content == 'all' or content.__contains__('optima'):
            # 根据查询优化
            chain.optima()
            logging.info("VRTree区块优化完成，交易VR*-tree节点数" + str(chain.tran_nodecount()))

            # 第二次交易查询
            begin = time.time()
            for mbr in mbrs:
                chain.query_tran(mbr)
            rtreea.append(time.time() - begin)
            rtreea_nodecount.append(chain.query_tran_node_count)
            logging.info("VRTree交易查询消耗（优化后）:" + str(rtreea[-1])+",访问节点数："+str(chain.query_tran_node_count))

        if content == 'all' or content.__contains__('blockdag'):
            # 创建block_dag区块(用于对比，来自https://github.com/ILDAR9/spatiotemporal blockdag.)
            logging.info('创建BlockDAG...')
            settings = dict(repeat_times=1, tr=60, D=3, bs=blocksize, alpha=10)
            settings.update(region_params)
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
            logging.info("BlockDAG执行查询...")
            begin = time.time()
            for i,query_range in enumerate(query_ranges):
                min_point, max_point, t_start, t_end = query_range['minp'],query_range['maxp'],query_range['t_start'],query_range['t_end']
                min_point, max_point = analysis_utils.__to_Cartesian_rect(min_point, max_point)
                rng = analysis_utils.__measure_time(sob.kd_range, 1, block_dag, min_point, max_point, t_start, t_end)

            kdtree.append(time.time() - begin)
            logging.info("BlockDAG交易查询消耗:" + str(kdtree[-1]))

    return rtreep,rtreep_nodecount,rtreea,rtreea_nodecount,kdtree
    #plt.plot(blocksizes,rtreep,color='blue')
    #plt.plot(blocksizes, rtreea,color='red')
    #plt.plot(blocksizes,kdtree,color='black')

def run_query_transaction(context,count=10,blocksizes=None,content='all',region_params=None):
    rtreep, rtreep_nodecount, rtreea, rtreea_nodecount, kdtree = [[]] * len(blocksizes), [[]] * len(blocksizes), [
        []] * len(blocksizes), [[]] * len(blocksizes), [[]] * len(blocksizes)
    for i in range(10):
        rp, rpnode, ra, ranode, kd = query_transaction(context, blocksizes,content,region_params)
        for j in range(blocksizes):
            rtreep[j] = rtreep[j] + [rp[j]]
            rtreep_nodecount[j] = rtreep_nodecount[j] + [rpnode[j]]
            rtreea[j] = rtreea[j] + [rp[j]]
            rtreea_nodecount[j] = rtreea_nodecount[j] + [ranode[j]]
            kdtree[j] = kdtree[j] + [kd[j]]

    rtreep = [np.average(e) for e in rtreep]
    rtreep_nodecount = [np.average(e) for e in rtreep_nodecount]
    rtreea = [np.average(e) for e in rtreea]
    rtreea_nodecount = [np.average(e) for e in rtreea_nodecount]
    kdtree = [np.average(e) for e in kdtree]

    logging.info("VRTree交易查询消耗（优化前）:" + str(rtreep))
    logging.info("VRTree交易访问节点数（优化前）:" + str(rtreep_nodecount))
    logging.info("VRTree交易查询消耗（优化前）:" + str(rtreea))
    logging.info("VRTree交易访问节点数（优化前）:" + str(rtreea_nodecount))
    logging.info("BlockDAG交易查询消耗:" + str(kdtree))

    return rtreep,rtreep_nodecount,rtreea,rtreea_nodecount,kdtree

def experiment1():
    context = Configuration(max_children_num=32, max_entries_num=8, account_length=8, account_count=200,
                            select_nodes_func='', merge_nodes_func='', split_node_func='')
    blocksizes = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]

    rtreep, rtreep_nodecount, rtreea, rtreea_nodecount, kdtree = run_query_transaction(context, count=3,
                                                                                       blocksizes=blocksizes,
                                                                                       content='all',
                                                                                       region_params=None)
    plt.figure(1)
    plt.plot(blocksizes, rtreep, color='blue')
    plt.plot(blocksizes, rtreea, color='red')
    plt.plot(blocksizes, kdtree, color='black')
    plt.legend()

    plt.figure(2)
    plt.plot(blocksizes, rtreep_nodecount, color='blue')
    plt.plot(blocksizes, rtreea_nodecount, color='red')
    plt.legend()

def experiment2():
    context = Configuration(max_children_num=32, max_entries_num=8, account_length=8, account_count=200,
                            select_nodes_func='', merge_nodes_func='', split_node_func='')
    blocksizes = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]

    region_params = {'geotype_probs': [0.5, 0.0, 0.5], 'length_probs': [0.6, 0.3, 0.1],
                     'lengthcenters': [50., 100., 300.], 'lengthscales': [1., 1., 1.]}
    rtreep, rtreep_nodecount, rtreea, rtreea_nodecount, kdtree = run_query_transaction(context, count=3,
                                                                                       blocksizes=blocksizes,
                                                                                       content='all',
                                                                                       region_params=region_params)
    plt.figure(3)
    plt.plot(blocksizes, rtreep, color='blue')
    plt.plot(blocksizes, rtreea, color='red')
    plt.plot(blocksizes, kdtree, color='black')
    plt.legend()

    plt.figure(4)
    plt.plot(blocksizes, rtreep_nodecount, color='blue')
    plt.plot(blocksizes, rtreea_nodecount, color='red')
    plt.legend()


if __name__ == '__main__':
    experiment1()
    #experiment2()

