import logging
import math
import os

import time
from random import random
import csv
import io


import matplotlib.pyplot as plt

from log import LogHandler
from utils import Configuration, Collections


import query

# 读取数据
logging = LogHandler('e2')

def experiment2(count=1,
                blocksizes = [20,40,60,80,100,120,140,160],
                context = Configuration(max_children_num=8, max_entries_num=8, account_length=8, account_count=200,
                            select_nodes_func='', merge_nodes_func='', split_node_func=''),
                query_param = dict(count=200, sizes=[2, 2, 2], posrandom=100, lengthcenter=0.05, lengthscale=0.1),
                region_params={'geotype_probs': [0.8, 0.0, 0.2], 'length_probs': [0.5, 0.3, 0.2],
                               'lengthcenters': [0.001, 0.01, 0.05], 'lengthscales': [0.5, 0.5, 0.5]},
                fig='show,save',
                savename='experiment2_524'
                ):
    '''
        实现Verkle AR*-tree、Verkel R*-tree对于非点类型数据查询的性能比较
        :return:
        '''

    logging.info('experiment2...')


    rtreep, rtreep_nodecount, rtreea, rtreea_nodecount, kdtree, _ = query.run_query_transaction(context, count=count,
                                                                                          blocksizes=blocksizes,
                                                                                          content='rtree,optima,blockdag',
                                                                                          query_param=query_param,
                                                                                          region_params=region_params,
                                                                                          query_mbrs=[],
                                                                                          refused=True)

    log_path = savename + '.csv'
    file = open(log_path, 'w', encoding='utf-8', newline='')
    csv_writer = csv.writer(file)
    csv_writer.writerow(rtreep)
    csv_writer.writerow(rtreep_nodecount)
    csv_writer.writerow(rtreea)
    csv_writer.writerow(rtreea_nodecount)
    csv_writer.writerow(kdtree)
    file.close()

    plt.figure(1)
    plt.plot(blocksizes, rtreep, color='blue',label='Verkel R*tree')
    plt.plot(blocksizes, rtreea, color='red', label='Verkel AR*tree')
    plt.plot(blocksizes, kdtree, color='black', label='Merkel KDtree')
    plt.grid(which='major', axis='x', linewidth=0.75, linestyle='-', color='0.75', dashes=(15, 10))
    plt.grid(which='major', axis='y', linewidth=0.75, linestyle='-', color='0.75', dashes=(15, 10))
    plt.xlabel('Block Size')
    plt.ylabel('Time(s)')
    plt.legend(loc='best')
    plt.savefig(savename+'_time.png')

    plt.figure(2)
    plt.plot(blocksizes, rtreep_nodecount, color='blue', label='Verkel R*tree')
    plt.plot(blocksizes, rtreea_nodecount, color='red', label='Verkel AR*tree')
    plt.grid(which='major', axis='x', linewidth=0.75, linestyle='-', color='0.75', dashes=(15, 10))
    plt.grid(which='major', axis='y', linewidth=0.75, linestyle='-', color='0.75', dashes=(15, 10))
    plt.xlabel('Block Size')
    plt.ylabel('Number of Nodes')
    plt.legend(loc='best')
    plt.savefig(savename+'_count.png')

if __name__ == '__main__':
    experiment2()