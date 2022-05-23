import logging
import csv
import matplotlib.pyplot as plt
from log import LogHandler
from utils import Configuration, Collections


import query

# 读取数据
logging = LogHandler('e1')

def experiment1(count = 1,
                blocksizes = [20, 40, 60, 80, 100, 120, 140, 160],
                context=Configuration(max_children_num=8, max_entries_num=8, account_length=8, account_count=200,
                                      select_nodes_func='', merge_nodes_func='', split_node_func=''),
                query_param = dict(count=200, sizes=[2, 2, 2], posrandom=100, lengthcenter=0.05, lengthscale=0.1),
                fig='show,save',
                savename='experimenttemp1'
                ):
    '''
    实现Verkle AR*-tree、Verkel R*-tree、Merkle KD-tree，scan-time在不同区块尺寸下的查询性能比较
     以及Verkle AR*-tree、Verkel R*-tree的访问节点数比较
    :return:
    '''



    rtreep, rtreep_nodecount, rtreea, rtreea_nodecount, kdtree,scan = query.run_query_transaction(context, count=count,
                                                                                       blocksizes=blocksizes,
                                                                                       content='blockdag',
                                                                                       query_param=query_param,
                                                                                       region_params={},
                                                                                       query_mbrs=[],
                                                                                       refused=True)

    logging.info("rtreep="+str(rtreep))
    logging.info("rtreep_nodecount=" + str(rtreep_nodecount))
    logging.info("rtreea=" + str(rtreea))
    logging.info("rtreea_nodecount=" + str(rtreea_nodecount))
    logging.info("kdtree=" + str(kdtree))
    logging.info("scan=" + str(scan))


    log_path = savename + '.csv'
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
    plt.savefig(savename+'_time.png')

    plt.figure(2)
    plt.plot(blocksizes, rtreep_nodecount, color='blue',label='Verkel R*tree')
    plt.plot(blocksizes, rtreea_nodecount, color='red',label='Verkel AR*tree')
    plt.grid(which='major', axis='x', linewidth=0.75, linestyle='-', color='0.75', dashes=(15, 10))
    plt.grid(which='major', axis='y', linewidth=0.75, linestyle='-', color='0.75', dashes=(15, 10))
    plt.xlabel('Block Size')
    plt.ylabel('Number of Nodes')
    plt.legend(loc='best')
    plt.savefig(savename+'_count.png')

if __name__ == '__main__':
    experiment1()