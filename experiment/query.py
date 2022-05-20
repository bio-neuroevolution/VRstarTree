import logging
import math
import os

import time
from random import random
import csv
import io

import numpy as np
import matplotlib.pyplot as plt

import geo
from log import LogHandler



from blocks import BTransaction, BlockChain
from dataloader import PocemonLoader
#from merklePatriciaTree.patricia_tree import MerklePatriciaTree
from utils import Configuration, Collections
from wise import simulation
from wise.blockDAG import analysis_utils
from wise.blockDAG import search_on_blockDAG as sob

# 读取数据
logging = LogHandler('run')

def _initdata(context,region_params={}):
    logging.info("VRTree读取数据...")
    dataLoader = PocemonLoader()
    df = dataLoader.refresh()  # 读取数据
    # df = dataLoader.extend_df(df=df,repeat_times=1)   #扩大数据
    df = dataLoader.normalize(df)  # 归一化
    transactions = [BTransaction(row['type'], row['lon'], row['lat'], row['ts'], None) for index, row in df.iterrows()]
    if len(region_params) > 0:
        dataLoader.create_region(transactions,
                                 geotype_probs=region_params['geotype_probs'],
                                 length_probs=region_params['length_probs'],
                                 lengthcenters=region_params['lengthcenters'],
                                 lengthscales=region_params['lengthscales'])

    # 创建和分配账户
    account_name, account_dis = dataLoader.create_account_name(context.account_count, context.account_length)
    accs = np.random.choice(account_name, len(transactions))
    diss = [account_dis[account_name.index(a)] for a in accs]
    for i, tr in enumerate(transactions):
        tr.account = accs[i]
    return transactions

def query_transaction(context,blocksizes,content='all',query_param={},region_params={},query_mbrs = [],refused=True):
    """
    一次查询实验
    :param conetxt 配置信息
    :param blocksizes 区块大小列表
    :param region_params 生成立方体实验数据参数
    :param query_param   生成查询数据所需参数，如果query_mbrs有效，则该参数不用
    :param refused       在优化时是否使用访问频率
    """
    # 读取数据
    transactions = _initdata(context,region_params)


    rtreep,rtreep_nodecount,rtreea,rtreea_nodecount,kdtree,scan = [],[],[],[],[],[]
    for i,blocksize in enumerate(blocksizes):
        logging.info("blocksize=" + str(blocksize))
        # 创建查询分布
        if len(query_mbrs)<=0:
            query_mbrs = create_query(count=query_param['count'], sizes=query_param['sizes'], posrandom=query_param['posrandom'], lengthcenter=query_param['lengthcenter'], lengthscale=query_param['lengthscale'])

        if content == 'all' or content.__contains__('rtree'):
            logging.info("VRTree创建区块...")
            # 创建区块链
            chain  = BlockChain(context,blocksize,transactions=transactions)
            logging.info("VRTree创建区块完成，交易VR*-tree节点总数=" + str(chain.tran_nodecount()) + ",平均="+str(chain.tran_nodecount()/len(chain.blocks))+",深度="+str(chain.header.trantrieRoot.depth))

            # 执行查询
            begin = time.time()
            for mbr in query_mbrs:
                chain.query_tran(mbr)
            rtreep.append(time.time() - begin)
            rtreep_nodecount.append(chain.query_tran_node_count)
            logging.info("VRTree交易查询消耗（优化前）:" + str(rtreep[-1])+'，访问节点数：'+str(chain.query_tran_node_count))
        else:
            rtreep.append(0.)
            rtreep_nodecount.append(0)

        if content == 'all' or content.__contains__('optima'):
            # 根据查询优化
            chain.optima(refused=refused)
            logging.info("VRTree区块优化完成，交易VR*-tree节点总数=" + str(chain.tran_nodecount()) + ",平均=" + str(
                chain.tran_nodecount() / len(chain.blocks)) + ",深度=" + str(chain.header.trantrieRoot.depth))


            # 第二次交易查询
            begin = time.time()
            for mbr in query_mbrs:
                chain.query_tran(mbr)
            rtreea.append(time.time() - begin)
            rtreea_nodecount.append(chain.query_tran_node_count)
            logging.info("VRTree交易查询消耗（优化后）:" + str(rtreea[-1])+",访问节点数："+str(chain.query_tran_node_count))
        else:
            rtreea.append(0.)
            rtreea_nodecount.append(0)

        if content == 'all' or content.__contains__('blockdag'):
            # 创建block_dag区块(用于对比，来自https://github.com/ILDAR9/spatiotemporal blockdag.)
            logging.info('创建BlockDAG...')
            settings = dict(repeat_times=1, tr=60, D=3, bs=blocksize, alpha=10)
            settings.update(region_params)
            block_dag = simulation.GeneratorDAGchain.generate(**settings)
            logging.info("创建BlockDAG完成, depth="+str(block_dag.merkle_kd_trees[1].depth))

            # 创建查询
            query_ranges = []
            for i,mbr in enumerate(query_mbrs):
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
            result_size = 0
            for i,query_range in enumerate(query_ranges):
                min_point, max_point, t_start, t_end = query_range['minp'],query_range['maxp'],query_range['t_start'],query_range['t_end']
                min_point, max_point = analysis_utils.__to_Cartesian_rect(min_point, max_point,[t_start, t_end])
                rng = analysis_utils.__measure_time(sob.kd_range, 1, block_dag, min_point, max_point, t_start, t_end)
                result_size += rng.loc[0][1]

            kdtree.append(time.time() - begin)
            logging.info("BlockDAG交易查询消耗:" + str(kdtree[-1])+",size="+str(result_size))
        else:
            kdtree.append(0.)

        if content == 'all' or content.__contains__('scan'):
            logging.info("scan执行查询...")
            scancount = 0
            begin = time.time()
            for mbr in query_mbrs:
                scancount += len([tx for tx in transactions if tx.mbr.isOverlop(mbr)])
            scan.append(time.time() - begin)
            logging.info("scan交易查询消耗:" + str(scan[-1]))
        else:
            scan.append(0)


    return rtreep,rtreep_nodecount,rtreea,rtreea_nodecount,kdtree,scan
    #plt.plot(blocksizes,rtreep,color='blue')
    #plt.plot(blocksizes, rtreea,color='red')
    #plt.plot(blocksizes,kdtree,color='black')

def run_query_transaction(context,count=10,blocksizes=None,content='all',query_param={},region_params={},query_mbrs = [],refused=True):
    rtreep, rtreep_nodecount, rtreea, rtreea_nodecount, kdtree,scan = [[]] * len(blocksizes), [[]] * len(blocksizes), [
        []] * len(blocksizes), [[]] * len(blocksizes), [[]] * len(blocksizes),[[]] * len(blocksizes)
    for i in range(count):
        rp, rpnode, ra, ranode, kd,sc = query_transaction(context, blocksizes,content,query_param,region_params,query_mbrs)
        for j in range(len(blocksizes)):
            rtreep[j] = rtreep[j] + [rp[j]]
            rtreep_nodecount[j] = rtreep_nodecount[j] + [rpnode[j]]
            rtreea[j] = rtreea[j] + [ra[j]]
            rtreea_nodecount[j] = rtreea_nodecount[j] + [ranode[j]]
            kdtree[j] = kdtree[j] + [kd[j]]
            scan[j] = scan[j] + [sc[j]]

    rtreep = [np.average(e) for e in rtreep]
    rtreep_nodecount = [np.average(e) for e in rtreep_nodecount]
    rtreea = [np.average(e) for e in rtreea]
    rtreea_nodecount = [np.average(e) for e in rtreea_nodecount]
    kdtree = [np.average(e) for e in kdtree]
    scan = [np.average(e) for e in scan]

    return rtreep,rtreep_nodecount,rtreea,rtreea_nodecount,kdtree,scan


def create_query(count=1,sizes:list=[2,2,2],posrandom=100,lengthcenter=0.05,lengthscale=0.025):
    """
    创建满足一定分布的查询
    :param count  int 生成的查询数
    :param sizes  list 生成的查询的每个维的中心点个数
    :param posrandom int 中心点漂移的百分比
    :param lengthcenter float 查询窗口的宽度均值
    :param lengthscale float 查询窗口的宽度方差
    """
    if lengthscale == 0:
        # 表示在均匀分布上采样
        r = []
        length = lengthcenter
        for c in range(count):
            center = [np.random.random() for dimension in range(len(sizes))]
            mbr = geo.Rectangle(dimension=len(sizes))
            for j in range(len(sizes)):
                mbr.update(j, center[j] - length / 2, center[j] + length / 2)
            r.append(mbr)
        return r
    elif lengthscale < 0:
        r = []
        #网格采样
        dimension = len(sizes)
        unit = math.ceil(count ** (1/dimension))
        step = math.ceil(1./unit)
        for i in range(unit):
            for j in range(unit):
                for k in range(unit):
                    mbr = geo.Rectangle(dimension)
                    mbr.update(0, i * step, (i + 1) * step)
                    mbr.update(1, j * step, (j + 1) * step)
                    mbr.update(2, k * step, (k + 1) * step)
                    r.append(mbr)
        return r
    elif Collections.all(lambda x:x==1,sizes):
        # 中心点采样
        r = []
        center = [0.5,0.5,0.5]
        for i in range(count):
            length = np.random.normal(loc=lengthcenter, scale=lengthscale, size=1)
            if length <= 0: length = lengthcenter
            mbr = geo.Rectangle(3)
            mbr.update(0, 0.5 - length /2,  0.5 + length / 2)
            mbr.update(1, 0.5 - length / 2, 0.5 + length / 2)
            mbr.update(2, 0.5 - length / 2, 0.5 + length / 2)
            r.append(mbr)
        return r

    # 在多元高斯分布上采样
    ds = []
    for i in range(len(sizes)):
        interval = 1./(sizes[i]+1)
        d = [(j+1)*interval for j in range(sizes[i])]
        ds.append(d)


    r = []
    for c in range(count):
        center = [np.random.choice(d,1)[0] for i,d in enumerate(ds)]
        for i in range(len(center)):
            center[i] += (np.random.random() * 2 - 1) / posrandom
        length = np.random.normal(loc=lengthcenter,scale=lengthscale,size=1)
        if length <= 0:length = lengthcenter
        mbr = geo.Rectangle(dimension=len(sizes))
        for j in range(len(sizes)):
            mbr.update(j,center[j]-length/2,center[j]+length/2)
        r.append(mbr)
    return r












if __name__ == '__main__':
    pass
    #experiment1()
    #experiment2()
    #experiment3()
    #experiment4()

