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

def query_transaction(context,blocksizes,content='all',query_param={},region_params={},query_mbrs = []):
    """
    一次查询实验
    :param conetxt 配置信息
    :param blocksizes 区块大小列表
    :param region_params 生成立方体实验数据参数
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
            chain.optima(refused=True)
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
            for i,query_range in enumerate(query_ranges):
                min_point, max_point, t_start, t_end = query_range['minp'],query_range['maxp'],query_range['t_start'],query_range['t_end']
                min_point, max_point = analysis_utils.__to_Cartesian_rect(min_point, max_point)
                rng = analysis_utils.__measure_time(sob.kd_range, 1, block_dag, min_point, max_point, t_start, t_end)

            kdtree.append(time.time() - begin)
            logging.info("BlockDAG交易查询消耗:" + str(kdtree[-1]))
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

def run_query_transaction(context,count=10,blocksizes=None,content='all',query_param={},region_params={},query_mbrs = []):
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
        mbr = geo.Rectangle(dimension=len(sizes))
        for j in range(len(sizes)):
            mbr.update(j,center[j]-length/2,center[j]+length/2)
        r.append(mbr)
    return r

def experiment1():
    '''
    实现Verkle AR*-tree、Verkel R*-tree、Merkle KD-tree，scan-time在不同区块尺寸下的查询性能比较
     以及Verkle AR*-tree、Verkel R*-tree的访问节点数比较
    :return:
    '''
    context = Configuration(max_children_num=8, max_entries_num=8, account_length=8, account_count=200,
                            select_nodes_func='', merge_nodes_func='', split_node_func='')
    query_param = dict(count=200, sizes=[2, 2, 2], posrandom=100, lengthcenter=0.05, lengthscale=0.1)
    blocksizes = [30,50,70,90,110,130,150,170,190,210,230,250]

    rtreep, rtreep_nodecount, rtreea, rtreea_nodecount, kdtree,scan = run_query_transaction(context, count=1,
                                                                                       blocksizes=blocksizes,
                                                                                       content='all',
                                                                                       query_param=query_param,
                                                                                       region_params={})

    logging.info("rtreep="+str(rtreep))
    logging.info("rtreep_nodecount=" + str(rtreep_nodecount))
    logging.info("rtreea=" + str(rtreea))
    logging.info("rtreea_nodecount=" + str(rtreea_nodecount))
    logging.info("kdtree=" + str(kdtree))
    logging.info("scan=" + str(scan))


    log_path = 'experiment1.csv'
    file = open(log_path, 'w', encoding='utf-8', newline='')
    csv_writer = csv.writer(file)
    csv_writer.writerow(rtreep)
    csv_writer.writerow(rtreep_nodecount)
    csv_writer.writerow(rtreea)
    csv_writer.writerow(rtreea_nodecount)
    csv_writer.writerow(kdtree)
    csv_writer.writerow(scan)
    file.close()

    plt.figure(1)
    plt.plot(blocksizes, rtreep, color='blue',label='Verkel R*tree')
    plt.plot(blocksizes, rtreea, color='red',label='Verkel AR*tree')
    plt.plot(blocksizes, kdtree, color='black',label='Merkel KDtree')
    #plt.plot(blocksizes, scan, color='green',label='Scan')
    plt.legend(loc='best')
    plt.grid(which='major', axis='x', linewidth=0.75, linestyle='-', color='0.75', dashes=(15, 10))
    plt.grid(which='major', axis='y', linewidth=0.75, linestyle='-', color='0.75', dashes=(15, 10))
    plt.xlabel('Block Size')
    plt.ylabel('Time(s)')
    plt.savefig('expeiment1_time.png')

    plt.figure(2)
    plt.plot(blocksizes, rtreep_nodecount, color='blue',label='Verkel R*tree')
    plt.plot(blocksizes, rtreea_nodecount, color='red',label='Verkel AR*tree')
    plt.grid(which='major', axis='x', linewidth=0.75, linestyle='-', color='0.75', dashes=(15, 10))
    plt.grid(which='major', axis='y', linewidth=0.75, linestyle='-', color='0.75', dashes=(15, 10))
    plt.xlabel('Block Size')
    plt.ylabel('Number of Nodes')
    plt.legend(loc='best')
    plt.savefig('expeiment1_count.png')

def experiment2():
    '''
        实现Verkle AR*-tree、Verkel R*-tree对于非点类型数据查询的性能比较
        :return:
        '''
    context = Configuration(max_children_num=8, max_entries_num=8, account_length=8, account_count=200,
                            select_nodes_func='', merge_nodes_func='', split_node_func='')
    query_param = dict(count=200, sizes=[2, 2, 2], posrandom=100, lengthcenter=0.05, lengthscale=0.1)
    blocksizes = [30, 50, 70, 90, 110, 130, 150, 170,190,210,230,250]

    region_params = {'geotype_probs': [0.2, 0.0, 0.8], 'length_probs': [0.5, 0.3, 0.2],
                     'lengthcenters': [0.01, 0.05, 0.1], 'lengthscales': [1., 1., 1.]}
    rtreep, rtreep_nodecount, rtreea, rtreea_nodecount, kdtree,_ = run_query_transaction(context, count=1,
                                                                                       blocksizes=blocksizes,
                                                                                       content='all',
                                                                                       query_param=query_param,
                                                                                       region_params=region_params)

    log_path = 'experiment2.csv'
    file = open(log_path, 'w', encoding='utf-8', newline='')
    csv_writer = csv.writer(file)
    csv_writer.writerow(rtreep)
    csv_writer.writerow(rtreep_nodecount)
    csv_writer.writerow(rtreea)
    csv_writer.writerow(rtreea_nodecount)
    csv_writer.writerow(kdtree)
    file.close()

    plt.figure(3)
    #plt.plot(blocksizes, rtreep, color='blue',label='Verkel R*tree')
    plt.plot(blocksizes, rtreea, color='red',label='Verkel AR*tree')
    plt.plot(blocksizes, kdtree, color='black',label='Merkel KDtree')
    plt.grid(which='major', axis='x', linewidth=0.75, linestyle='-', color='0.75', dashes=(15, 10))
    plt.grid(which='major', axis='y', linewidth=0.75, linestyle='-', color='0.75', dashes=(15, 10))
    plt.xlabel('Block Size')
    plt.ylabel('Time(s)')
    plt.legend(loc='best')
    plt.savefig('experiment2_time.png')

    plt.figure(4)
    plt.plot(blocksizes, rtreep_nodecount, color='blue',label='Verkel R*tree')
    plt.plot(blocksizes, rtreea_nodecount, color='red',label='Verkel AR*tree')
    plt.grid(which='major', axis='x', linewidth=0.75, linestyle='-', color='0.75', dashes=(15, 10))
    plt.grid(which='major', axis='y', linewidth=0.75, linestyle='-', color='0.75', dashes=(15, 10))
    plt.xlabel('Block Size')
    plt.ylabel('Number of Nodes')
    plt.legend(loc='best')
    plt.savefig('experiment2_count.png')

def experiment3():
    '''
        实现Verkle AR*-tree、Verkel R*-tree在不同查询分布下的比较
        :return:
        '''
    max_children_nums = [4,8,16,32,48,64,80,96,108,128]
    blocksizes = [90]
    itercount = 1
    querycount = 5000
    region_params = {'geotype_probs': [0.2, 0.0, 0.8], 'length_probs': [0.5, 0.3, 0.2],
                     'lengthcenters': [0.01, 0.05, 0.1], 'lengthscales': [1., 1., 1.]}

    rtree_time_init = {'center':[],'gaussian':[],'uniform':[],'grid':[],'avg':[]}
    rtree_time_optima = {'center': [], 'gaussian': [], 'uniform': [], 'grid': [], 'avg': []}
    rtree_count_init = {'center': [], 'gaussian': [], 'uniform': [], 'grid': [], 'avg': []}
    rtree_count_optima = {'center': [], 'gaussian': [], 'uniform': [], 'grid': [],'avg':[]}

    query_param_dict = dict(
                        center = dict(count=querycount, sizes=[1,1,1], posrandom=100, lengthcenter=0.1, lengthscale=0.05),
                        gaussian = dict(count=querycount, sizes=[2,2,2], posrandom=100, lengthcenter=0.1, lengthscale=0.05),
                        uniform = dict(count=querycount, sizes=[2, 2, 2], posrandom=100, lengthcenter=0.1, lengthscale=0.0),
                        grid= dict(count=querycount, sizes=[2, 2, 2], posrandom=100, lengthcenter=0.05, lengthscale=-1.0)
                        )
    query_mbrs = {}
    for key, query_param in query_param_dict.items():
        mbrs = create_query(count=query_param['count'],sizes=query_param['sizes'],posrandom=query_param['posrandom'],lengthcenter=query_param['lengthcenter'],lengthscale=query_param['lengthscale'])
        query_mbrs[key] = mbrs

    for i,max_children_num in enumerate(max_children_nums):
        logging.info("max_children_num="+str(max_children_num))
        context = Configuration(max_children_num=max_children_num, max_entries_num=max_children_num, account_length=8, account_count=200,
                                select_nodes_func='', merge_nodes_func='', split_node_func='')

        for key,query_param in query_param_dict.items():
            logging.info('query with :' + key)
            r1, n1, r2, n2, _, _ = run_query_transaction(context, count=itercount,blocksizes=blocksizes,
                                                         content='rtree,optima',query_param=query_param,
                                                         region_params=region_params,query_mbrs=query_mbrs[key])

            rtree_time_init[key].append(r1[0])
            rtree_time_optima[key].append(r2[0])
            rtree_count_init[key].append(n1[0])
            rtree_count_optima[key].append(n2[0])

    for key, query_param in query_param_dict.items():
        logging.info(key)
        logging.info(str(rtree_time_init[key]))
        logging.info(str(rtree_time_optima[key]))
        logging.info(str(rtree_count_init[key]))
        logging.info(str(rtree_count_optima[key]))



    log_path = 'experiment3.csv'
    file = open(log_path, 'w', encoding='utf-8', newline='')
    csv_writer = csv.writer(file)
    for key, query_param in query_param_dict.items():
        csv_writer.writerow(key)
        csv_writer.writerow(rtree_time_init[key])
        csv_writer.writerow(rtree_time_optima[key])
        csv_writer.writerow(rtree_count_init[key])
        csv_writer.writerow(rtree_count_optima[key])
    file.close()


    plt.figure(5)
    plt.plot(max_children_nums, rtree_time_optima['center'], color='blue',label="center")
    plt.plot(max_children_nums, rtree_time_optima['gaussian'], color='red', label="gaussian")
    plt.plot(max_children_nums, rtree_time_optima['uniform'], color='green',label="uniform")
    plt.grid(which='major', axis='x', linewidth=0.75, linestyle='-', color='0.75', dashes=(15, 10))
    plt.grid(which='major', axis='y', linewidth=0.75, linestyle='-', color='0.75', dashes=(15, 10))
    plt.xlabel('max number of child nodes')
    plt.ylabel('Time(s)')
    #plt.plot(max_children_nums, rtree_time_optima['grid'], color='black', label="grid")
    plt.legend(loc='best')
    plt.savefig('experiment3_time.png')

    plt.figure(6)
    plt.plot(max_children_nums, rtree_count_optima['center'], color='blue', label="center")
    plt.plot(max_children_nums, rtree_count_optima['gaussian'], color='red', label="gaussian")
    plt.plot(max_children_nums, rtree_count_optima['uniform'], color='green', label="uniform")
    #plt.plot(max_children_nums, rtree_count_optima['grid'], color='black', label="grid")
    plt.grid(which='major', axis='x', linewidth=0.75, linestyle='-', color='0.75', dashes=(15, 10))
    plt.grid(which='major', axis='y', linewidth=0.75, linestyle='-', color='0.75', dashes=(15, 10))
    plt.xlabel('max number of child nodes')
    plt.ylabel('Number of Nodes')
    plt.legend(loc="best")
    plt.savefig('experiment3_count.png')

def experiment4():
    """
    比较MPT树和Verkle树，Merkle KD-Tree和两个Verkle AR*-tree树的证明长度
    """
    context = Configuration(max_children_num=8, max_entries_num=8, account_length=8, account_count=200,
                            select_nodes_func='', merge_nodes_func='', split_node_func='')
    query_param = dict(count=200, sizes=[2, 2, 2], posrandom=100, lengthcenter=0.05, lengthscale=0.1)
    blocksizes = [30, 50, 70, 90, 110, 130, 150, 170, 190, 210, 230, 250]
    account_lengths,tran_lengths,traj_lengths,block_dag_lengths,mpt_lengths = [],[],[],[],[]

    proof_length_unit ,proof_sample_count= 64,10
    # 读取数据
    transactions = _initdata(context, {})

    for i,blocksize in enumerate(blocksizes):
        # 创建区块链
        logging.info('创建BlockChain...')
        chain = BlockChain(context, blocksize, transactions=transactions)
        # 计算叶节点的证明长度
        account_length,tran_length,traj_length = chain.proof_length(count=proof_sample_count,unit=proof_length_unit)
        print(str(account_length)+","+str(tran_length)+","+str(traj_length))

        logging.info('创建BlockDAG...')
        settings = dict(repeat_times=1, tr=60, D=3, bs=blocksize, alpha=10)
        block_dag = simulation.GeneratorDAGchain.generate(**settings)
        block_dag_length = block_dag.merkle_kd_trees[1].proof_length(proof_length_unit)
        print(block_dag_length)


        logging.info('创建MPT...')
        dataLoader = PocemonLoader()
        account_names, _ = dataLoader.create_account_name(blocksize, context.account_length)
        mpt = None #MerklePatriciaTree(xh=i+1,from_scratch=True)

        for name in account_names:
            mpt.insert(name)
        mpt_length = mpt.proof_length(count=proof_sample_count,unit=proof_length_unit)
        print(mpt_length)
        #import shutil
        #shutil.rmtree('./db')

        account_lengths.append(account_length)
        tran_lengths.append(tran_length)
        traj_lengths.append(tran_length)
        block_dag_lengths.append(block_dag_length)
        mpt_lengths.append(mpt_length)

    logging.info('实验四结果：')
    logging.info(blocksizes)
    logging.info(account_lengths)
    logging.info(tran_lengths)
    logging.info(traj_lengths)
    logging.info(block_dag_lengths)
    logging.info(mpt_lengths)

    log_path = 'experiment4.csv'
    file = open(log_path, 'w', encoding='utf-8', newline='')
    csv_writer = csv.writer(file)
    csv_writer.writerow(blocksizes)
    csv_writer.writerow(account_lengths)
    csv_writer.writerow(tran_lengths)
    csv_writer.writerow(traj_lengths)
    csv_writer.writerow(block_dag_lengths)
    csv_writer.writerow(mpt_lengths)
    file.close()

    plt.figure(7)
    plt.plot(blocksizes, account_lengths,label="Account")
    plt.plot(blocksizes, tran_lengths,label="Transaction")
    plt.plot(blocksizes, traj_lengths,label="Trajectory")
    plt.plot(blocksizes, block_dag_lengths,label='BlockDAG')
    plt.plot(blocksizes, mpt_lengths,label='MPT')
    plt.grid(which='major', axis='x', linewidth=0.75, linestyle='-', color='0.75', dashes=(15, 10))
    plt.grid(which='major', axis='y', linewidth=0.75, linestyle='-', color='0.75', dashes=(15, 10))
    plt.xlabel('Block size')
    plt.ylabel('Proof length')
    plt.legend(loc='best')
    plt.savefig('experiment4_proof.png')





if __name__ == '__main__':
    #experiment1()
    experiment2()
    #experiment3()
    #experiment4()

