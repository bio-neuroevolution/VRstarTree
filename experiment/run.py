import time
from random import random

import numpy as np
import matplotlib.pyplot as plt

from log import LogHandler



from blocks import BTransaction, BlockChain
from dataloader import PocemonLoader
from merklePatriciaTree.patricia_tree import MerklePatriciaTree
from utils import Configuration
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

def query_transaction(context,blocksizes,content='all',query_param={},region_params={}):
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

        mbrs = BlockChain.create_query(count=query_param['count'], sizes=query_param['sizes'], posrandom=query_param['posrandom'], lengthcenter=query_param['lengthcenter'], lengthscale=query_param['lengthscale'])

        if content == 'all' or content.__contains__('rtree'):
            logging.info("VRTree创建区块...")
            # 创建区块链
            chain  = BlockChain(context,blocksize,transactions=transactions)
            logging.info("VRTree创建区块完成，交易VR*-tree节点总数=" + str(chain.tran_nodecount()) + ",平均="+str(chain.tran_nodecount()/len(chain.blocks))+",深度="+str(chain.header.trantrieRoot.depth))

            # 执行查询
            begin = time.time()
            for mbr in mbrs:
                chain.query_tran(mbr)
            rtreep.append(time.time() - begin)
            rtreep_nodecount.append(chain.query_tran_node_count)
            logging.info("VRTree交易查询消耗（优化前）:" + str(rtreep[-1])+'，访问节点数：'+str(chain.query_tran_node_count))

        if content == 'all' or content.__contains__('optima'):
            # 根据查询优化
            chain.optima()
            logging.info("VRTree区块优化完成，交易VR*-tree节点总数=" + str(chain.tran_nodecount()) + ",平均=" + str(
                chain.tran_nodecount() / len(chain.blocks)) + ",深度=" + str(chain.header.trantrieRoot.depth))


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

        if content == 'all' or content.__contains__('scan'):
            logging.info("scan执行查询...")
            scancount = 0
            begin = time.time()
            for mbr in mbrs:
                scancount += len([tx for tx in transactions if tx.mbr.isOverlop(mbr)])
            scan.append(time.time() - begin)
            logging.info("scan交易查询消耗:" + str(scan[-1]))


    return rtreep,rtreep_nodecount,rtreea,rtreea_nodecount,kdtree,scan
    #plt.plot(blocksizes,rtreep,color='blue')
    #plt.plot(blocksizes, rtreea,color='red')
    #plt.plot(blocksizes,kdtree,color='black')

def run_query_transaction(context,count=10,blocksizes=None,content='all',query_param={},region_params={}):
    rtreep, rtreep_nodecount, rtreea, rtreea_nodecount, kdtree,scan = [[]] * len(blocksizes), [[]] * len(blocksizes), [
        []] * len(blocksizes), [[]] * len(blocksizes), [[]] * len(blocksizes),[[]] * len(blocksizes)
    for i in range(count):
        rp, rpnode, ra, ranode, kd,sc = query_transaction(context, blocksizes,content,query_param,region_params)
        for j in range(len(blocksizes)):
            rtreep[j] = rtreep[j] + [rp[j]]
            rtreep_nodecount[j] = rtreep_nodecount[j] + [rpnode[j]]
            rtreea[j] = rtreea[j] + [rp[j]]
            rtreea_nodecount[j] = rtreea_nodecount[j] + [ranode[j]]
            kdtree[j] = kdtree[j] + [kd[j]]
            scan[i] = scan[j] + [sc[j]]

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

    return rtreep,rtreep_nodecount,rtreea,rtreea_nodecount,kdtree,scan

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

    rtreep, rtreep_nodecount, rtreea, rtreea_nodecount, kdtree,scan = run_query_transaction(context, count=2,
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

    plt.figure(1)
    plt.plot(blocksizes, rtreep, color='blue',label='Verkel R*tree')
    plt.plot(blocksizes, rtreea, color='red',label='Verkel AR*tree')
    plt.plot(blocksizes, kdtree, color='black',label='Merkel KDtree')
    plt.plot(blocksizes, scan, color='green',label='Scan')
    plt.legend(loc='best')
    plt.savefig('expeiment1_time.png')

    plt.figure(2)
    plt.plot(blocksizes, rtreep_nodecount, color='blue',label='Verkel R*tree')
    plt.plot(blocksizes, rtreea_nodecount, color='red',label='Verkel AR*tree')
    plt.legend(loc='best')
    plt.savefig('expeiment1_count.png')

def experiment2():
    '''
        实现Verkle AR*-tree、Verkel R*-tree对于非点类型数据查询的性能比较
        :return:
        '''
    context = Configuration(max_children_num=32, max_entries_num=8, account_length=8, account_count=200,
                            select_nodes_func='', merge_nodes_func='', split_node_func='')
    query_param = dict(count=200, sizes=[2, 2, 2], posrandom=100, lengthcenter=0.05, lengthscale=0.1)
    blocksizes = [30, 50, 70, 90, 110, 130, 150, 170, 190]

    region_params = {'geotype_probs': [0.5, 0.0, 0.5], 'length_probs': [0.6, 0.3, 0.1],
                     'lengthcenters': [50., 100., 300.], 'lengthscales': [1., 1., 1.]}
    rtreep, rtreep_nodecount, rtreea, rtreea_nodecount, kdtree = run_query_transaction(context, count=3,
                                                                                       blocksizes=blocksizes,
                                                                                       content='all',
                                                                                       query_param=query_param,
                                                                                       region_params=region_params)
    plt.figure(3)
    plt.plot(blocksizes, rtreep, color='blue',label='Verkel R*tree')
    plt.plot(blocksizes, rtreea, color='red',label='Verkel AR*tree')
    plt.plot(blocksizes, kdtree, color='black',label='Merkel KDtree')
    plt.legend(loc='best')
    plt.savefig('experiment2_time.png')

    plt.figure(4)
    plt.plot(blocksizes, rtreep_nodecount, color='blue',label='Verkel R*tree')
    plt.plot(blocksizes, rtreea_nodecount, color='red',label='Verkel AR*tree')
    plt.legend(loc='best')
    plt.savefig('experiment2_count.png')

def experiment3():
    '''
        实现Verkle AR*-tree、Verkel R*-tree在不同查询分布下的比较
        :return:
        '''
    max_children_nums = [2,4,8,16,32,48,64,80,96]
    blocksizes = [90]
    itercount = 2
    rtree_gaussian_time,rtree_gaussian_count,rtree_uniform_time,rtree_uniform_count = [],[],[],[]
    for i,max_children_num in enumerate(max_children_nums):
        context = Configuration(max_children_num=max_children_num, max_entries_num=max_children_num, account_length=8, account_count=200,
                                select_nodes_func='', merge_nodes_func='', split_node_func='')
        query_param = dict(count=200, sizes=[2, 2, 2], posrandom=100, lengthcenter=0.05, lengthscale=0.1)

        rtreep1, rtreep_nodecount1, rtreea1, rtreea_nodecount1, _, _ = run_query_transaction(context, count=itercount,
                                                                                             blocksizes=blocksizes,
                                                                                             content='rtree,optima',
                                                                                             query_param=query_param,
                                                                                             region_params={})
        rtree_gaussian_time.append(rtreep1[0])
        rtree_gaussian_count.append(rtreep_nodecount1[0])


        query_param = dict(count=200, sizes=[], posrandom=0, lengthcenter=0.05, lengthscale=0)
        rtreep2, rtreep_nodecount2, rtreea2, rtreea_nodecount2, _, _ = run_query_transaction(context, count=itercount,
                                                                                             blocksizes=blocksizes,
                                                                                             content='rtree,optima',
                                                                                             query_param=query_param,
                                                                                             region_params={})
        rtree_uniform_time.append(rtreep2[0])
        rtree_uniform_count.append(rtreea_nodecount2[0])


    logging.info("rtree time(gaussian)=" + str(rtree_gaussian_time))
    logging.info("rtree count(gaussian)=" + str(rtree_gaussian_count))
    logging.info("rtree time(uniform)=" + str(rtree_uniform_time))
    logging.info("rtree count(uniform)=" + str(rtree_uniform_count))


    plt.figure(5)
    plt.plot(max_children_nums, rtree_gaussian_time, color='blue',label="Gaussian")
    plt.plot(max_children_nums, rtree_uniform_time, color='red',label="Uniform")
    plt.legend(loc='best')
    plt.savefig('experiment3_time.png')

    plt.figure(6)
    plt.plot(max_children_nums, rtree_gaussian_count, color='blue',label="Gaussian")
    plt.plot(max_children_nums, rtree_uniform_count, color='red',label="Uniform")
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
        block_dag_length = block_dag.merkle_kd_trees.proof_length(proof_length_unit)
        print(block_dag_length)


        logging.info('创建MPT...')
        dataLoader = PocemonLoader()
        account_names, _ = dataLoader.create_account_name(blocksize, context.account_length)
        mpt = MerklePatriciaTree()

        for name in account_names:
            mpt.insert(name)
        mpt_length = mpt.proof_length(count=proof_sample_count,unit=proof_length_unit)
        print(mpt_length)

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
    plt.figure(7)
    plt.plot(blocksizes, account_lengths,label="Account")
    plt.plot(blocksizes, tran_lengths,label="Transaction")
    plt.plot(blocksizes, traj_lengths,label="Trajectory")
    plt.plot(blocksizes, block_dag_lengths,label='BlockDAG')
    plt.plot(blocksizes, mpt_lengths,label='MPT')
    plt.legend(loc='best')
    plt.savefig('experiment4_proof.png')



def fun(matrix : list) -> list:
    """
    将图的邻接矩阵转换为边构成的元组列表
    :param   matrix list[[int]]  邻接矩阵
    :result  list[tuple] 每个边构成的列表
    :example
              matrix = [[0,1,0,0],
                        [0,0,1,1],
                        [0,0,0,0],
                        [0,0,0,0]
                       ]
              result = fun(matrix)
              print(str(result))   #  [('0','1'),('1','2'),('1','3')]
    """
    result = []
    for rowno,row in enumerate(matrix):
        for colno,col in enumerate(row):
            if col != 1:
                continue
            result.append((str(rowno),str(colno)))
    return result

if __name__ == '__main__':
    #experiment1()
    #experiment2()
    #experiment3()
    experiment4()

