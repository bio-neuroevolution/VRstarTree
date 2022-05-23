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

import query

logging = LogHandler('e3')

def experiment3(count = 1,
                query_param_dict = dict(
                        center = dict(count=6000, sizes=[1,1,1], posrandom=100, lengthcenter=0.1, lengthscale=0.05),
                        gaussian = dict(count=6000, sizes=[2,2,2], posrandom=100, lengthcenter=0.1, lengthscale=0.05),
                        uniform = dict(count=6000, sizes=[2, 2, 2], posrandom=100, lengthcenter=0.1, lengthscale=0.0),
                        grid= dict(count=6000, sizes=[2, 2, 2], posrandom=100, lengthcenter=0.1, lengthscale=-1.0)
                        ),
                max_children_nums = [4,8,16,24,32,40,48,64,72,80,88,96],
                blocksizes = [80],
                fig='show,save',
                savename='experiment3',
                region_params={}):
    '''
        实现Verkle AR*-tree、Verkel R*-tree在不同查询分布下的比较
      :return:
    '''
    logging.info('experiment3...')
    if not savename:
        savename = 'experiment3'
    itercount = count
    #region_params = {'geotype_probs': [0.9, 0.0, 0.1], 'length_probs': [0.6, 0.3, 0.1],
    #                 'lengthcenters': [0.001, 0.005, 0.01], 'lengthscales': [0.05, 0.05, 0.05]}


    rtree_time = {'center':[],'gaussian':[],'uniform':[],'grid':[],'nonfre':[]}
    rtree_count = {'center': [], 'gaussian': [], 'uniform': [], 'grid': [], 'nonfre': []}


    query_mbrs = {}
    for key, query_param in query_param_dict.items():
        mbrs = query.create_query(count=query_param['count'],sizes=query_param['sizes'],posrandom=query_param['posrandom'],lengthcenter=query_param['lengthcenter'],lengthscale=query_param['lengthscale'])
        query_mbrs[key] = mbrs

    for i,max_children_num in enumerate(max_children_nums):
        logging.info(max_children_num=+str(max_children_num))
        context = Configuration(max_children_num=max_children_num, max_entries_num=max_children_num, account_length=8, account_count=200,
                                select_nodes_func='', merge_nodes_func='', split_node_func='')

        for key,query_param in query_param_dict.items():
            logging.info('query with :' + key)
            r11, n11, r12, n12, _, _ = query.run_query_transaction(context, count=itercount,blocksizes=blocksizes,
                                                         content='rtree,optima',query_param=query_param,
                                                         region_params=region_params,query_mbrs=query_mbrs[key],refused=False)


            r21, n21, r22, n22, _, _ = query.run_query_transaction(context, count=itercount, blocksizes=blocksizes,
                                                             content='rtree,optima', query_param=query_param,
                                                             region_params=region_params, query_mbrs=query_mbrs[key],
                                                             refused=True)
            rtree_time[key].append([r11,r12,r21,r22])
            rtree_count[key].append([n11,n12,n21,n22])


    for key, query_param in query_param_dict.items():
        logging.info(key)
        logging.info(str(rtree_time[key]))
        logging.info(str(rtree_count[key]))


    log_path = savename + '.csv'
    file = open(log_path, 'w', encoding='utf-8', newline='')
    csv_writer = csv.writer(file)
    csv_writer.writerow(max_children_nums)
    for key, query_param in query_param_dict.items():
        csv_writer.writerow(key)
        csv_writer.writerow(rtree_time[key])
        csv_writer.writerow(rtree_count[key])
    file.close()

    colors = ['black','blue','black','red']
    labels = [key+'-R*1',key+'-Non-ref',key+'-R*2',key+'-Ref']

    count = 1
    for key in query_param_dict.keys():
        values = rtree_time[key]
        values = [[d[0][0] if d[0][0] != 0 else 1, d[1][0], d[2][0] if d[2][0] != 0 else 1, d[3][0]] for d in values]
        plt.subplot(int('24' + str(count)))
        count += 1
        norefs = [(d[0] - d[1])/d[0] for d in values]
        refs = [(d[2] - d[3])/d[2] for d in values]
        print('norefs-time='+ str(norefs))
        print('refs-time='+str(refs))

        plt.plot(max_children_nums, norefs, color='blue', label=key + _noref)
        plt.plot(max_children_nums, refs, color='red', label=key + _ref)
        plt.grid(which='major', axis='x', linewidth=0.75, linestyle='-', color='0.75', dashes=(15, 10))
        plt.grid(which='major', axis='y', linewidth=0.75, linestyle='-', color='0.75', dashes=(15, 10))
        plt.legend(loc='best')
        if count == 2:
            plt.xlabel('max children number of node')
            plt.ylabel('The optimized ratio of time')

    for key in query_param_dict.keys():
        values = rtree_count[key]
        values = [[d[0][0] if d[0][0] != 0 else 1, d[1][0], d[2][0] if d[2][0] != 0 else 1, d[3][0]] for d in values]
        plt.subplot(int('24' + str(count)))
        count += 1
        norefs = [(d[0] - d[1])/d[0] for d in values]
        refs = [(d[2] - d[3])/d[2] for d in values]

        print('norefs-count=' + str(norefs))
        print('refs-count=' + str(refs))

        plt.plot(max_children_nums, norefs, color='blue', label=key + _noref)
        plt.plot(max_children_nums, refs, color='red', label=key + _ref)
        plt.grid(which='major', axis='x', linewidth=0.75, linestyle='-', color='0.75', dashes=(15, 10))
        plt.grid(which='major', axis='y', linewidth=0.75, linestyle='-', color='0.75', dashes=(15, 10))
        plt.legend(loc='best')
        if count == 6:
            plt.xlabel('max children num of node')
            plt.ylabel('The optimized ratio of nodes')

    if fig.__contains__('save'):
        plt.savefig(savename+'.png')
    if fig.__contains__('show'):
        plt.show()




if __name__ == '__main__':
    experiment3()
