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
                        center = dict(count=6000, sizes=[1,1,1], posrandom=100, lengthcenter=0.3, lengthscale=0.05),
                        gaussian = dict(count=6000, sizes=[2,2,2], posrandom=100, lengthcenter=0.1, lengthscale=0.05),
                        uniform = dict(count=6000, sizes=[2, 2, 2], posrandom=100, lengthcenter=0.1, lengthscale=0.0),
                        grid= dict(count=6000, sizes=[2, 2, 2], posrandom=100, lengthcenter=0.05, lengthscale=-1.0)
                        ),
                max_children_nums = [2,4,8,16,32,48,64,80,96],
                blocksizes = [80],
                fig='show,save',
                figname='experiment31.png',
                region_params={}):
    '''
        实现Verkle AR*-tree、Verkel R*-tree在不同查询分布下的比较
      :return:
    '''
    logging.info('experiment3...')

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
        logging.info("max_children_num="+str(max_children_num))
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


    log_path = 'experiment3.csv'
    file = open(log_path, 'w', encoding='utf-8', newline='')
    csv_writer = csv.writer(file)
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
        plt.subplot(int('24' + str(count)))
        count += 1
        norefs = [(d[0][0] - d[1][0])/d[0][0] for d in values]
        refs = [(d[2][0] - d[3][0])/d[2][0] for d in values]
        print('norefs-time='+ str(norefs))
        print('refs-time='+str(refs))

        plt.plot(max_children_nums, norefs, color='blue', label=key + "_noref")
        plt.plot(max_children_nums, refs, color='red', label=key + "_ref")
        plt.grid(which='major', axis='x', linewidth=0.75, linestyle='-', color='0.75', dashes=(15, 10))
        plt.grid(which='major', axis='y', linewidth=0.75, linestyle='-', color='0.75', dashes=(15, 10))
        plt.xlabel('max number of child nodes')
        plt.ylabel('Time(s)')
        plt.legend(loc='best')

    for key in query_param_dict.keys():
        values = rtree_count[key]
        plt.subplot(int('24' + str(count)))
        count += 1
        norefs = [(d[0][0] - d[1][0])/d[0][0] for d in values]
        refs = [(d[2][0] - d[3][0])/d[2][0] for d in values]

        print('norefs-count=' + str(norefs))
        print('refs-count=' + str(refs))

        plt.plot(max_children_nums, norefs, color='blue', label=key + "_noref")
        plt.plot(max_children_nums, refs, color='red', label=key + "_ref")
        plt.grid(which='major', axis='x', linewidth=0.75, linestyle='-', color='0.75', dashes=(15, 10))
        plt.grid(which='major', axis='y', linewidth=0.75, linestyle='-', color='0.75', dashes=(15, 10))
        plt.xlabel('max number of child nodes')
        plt.ylabel('number of nodes')
        plt.legend(loc='best')

    if fig.__contains__('save'):
        plt.savefig(figname)
    if fig.__contains__('show'):
        plt.show()

def drawfig():

    results_time = dict(center_time=
                        [[[2.714740753173828], [2.5531692504882812], [2.5431666374206543], [2.460423231124878]],[[2.4394726753234863], [2.50034499168396], [3.0258774757385254], [2.459398031234741]],[[2.341735363006592], [2.4644415378570557], [2.3636810779571533], [2.4444942474365234]],[[2.4115841388702393], [2.712719678878784], [2.4105498790740967], [2.4683945178985596]],[[2.492338180541992], [2.5062661170959473], [2.450448989868164], [2.480369806289673]],[[2.464374303817749], [2.436460018157959], [2.4504787921905518], [2.5182342529296875]],[[2.4045722484588623], [2.510289192199707], [2.4105541706085205], [3.255270481109619]],[[2.5960891246795654], [2.608031988143921], [2.586087703704834], [2.742668628692627]],[[2.6020450592041016], [2.6150102615356445], [2.623957872390747], [2.609994649887085]],[[2.599080801010132], [2.6070609092712402], [3.196424961090088], [2.622986316680908]]],
                        gausian_time=
                        [[[9.3569815158844], [5.388627052307129], [9.648204565048218], [5.400564193725586]],[[5.6827850341796875], [3.645249128341675], [5.566150903701782], [3.9045350551605225]],[[4.3174543380737305], [3.115664482116699], [4.369286775588989], [3.087717294692993]],[[3.74997615814209], [2.985049247741699], [3.7330524921417236], [2.941103458404541]],[[3.361016273498535], [2.9820003509521484], [3.6641743183135986], [2.944100856781006]],[[3.3141696453094482], [3.2782399654388428], [3.6013731956481934], [2.971029043197632]],[[3.240344285964966], [3.0817933082580566], [3.2203919887542725], [3.0648081302642822]],[[3.2293689250946045], [3.3380744457244873], [3.585416316986084], [3.32511043548584]],[[3.207428216934204], [3.33608341217041], [3.2363200187683105], [3.330092191696167]],[[3.2273447513580322], [3.3340871334075928], [3.2084579467773438], [3.3001503944396973]]],
                        uniform_time=
                        [[[9.664138793945312], [4.13392186164856], [9.381946325302124], [4.145917654037476]],[[5.182113885879517], [2.6199586391448975], [5.186136484146118], [2.6838204860687256]],[[4.006319522857666], [2.255964756011963], [4.011277437210083], [2.347752094268799]],[[3.215432643890381], [2.6928300857543945], [3.200474977493286], [2.278881788253784]],[[2.9301674365997314], [2.282891035079956], [2.9181997776031494], [2.337751626968384]],[[2.851377010345459], [2.391606330871582], [2.80849289894104], [2.451446533203125]],[[2.774580955505371], [2.555171251296997], [2.7516725063323975], [2.6210215091705322]],[[2.7585973739624023], [2.689810037612915], [2.7297022342681885], [2.7176966667175293]],[[2.800541877746582], [2.718761682510376], [2.7177352905273438], [3.018953323364258]],[[2.7516446113586426], [2.6908013820648193], [2.7456605434417725], [2.735687255859375]]],
                        grid_time=
                        [[[1.164886713027954], [1.169844627380371], [1.163917064666748], [1.1808063983917236]],[[0.7839035987854004], [0.8188116550445557], [0.7769231796264648], [0.909569263458252]],[[0.6662473678588867], [0.6961324214935303], [0.6721696853637695], [0.6911194324493408]],[[0.6991550922393799], [0.6053814888000488], [0.6053752899169922], [0.6283490657806396]],[[0.5694782733917236], [0.5784258842468262], [0.5705032348632812], [0.5864322185516357]],[[0.5614993572235107], [0.5684454441070557], [0.564490795135498], [0.5734670162200928]],[[0.5595335960388184], [0.5745007991790771], [0.5525219440460205], [0.5595321655273438]],[[0.5415525436401367], [0.5524945259094238], [0.5425493717193604], [0.5515265464782715]],[[0.5455420017242432], [0.5575103759765625], [0.5435550212860107], [0.5535497665405273]],[[0.541522741317749], [0.5724694728851318], [0.7370295524597168], [0.5654876232147217]]]
                        )
    results_count = dict(
                        center_count=
                        [[[7.0], [11.0], [7.0], [4.0]],[[1.0], [8.0], [1.0], [7.0]],[[0.0], [10.0], [0.0], [6.0]],[[0.0], [8.0], [0.0], [12.0]],[[20.0], [21.0], [20.0], [22.0]],[[37.0], [21.0], [37.0], [36.0]],[[21.0], [35.0], [21.0], [66.0]],[[80.0], [69.0], [80.0], [69.0]],[[80.0], [69.0], [80.0], [69.0]],[[80.0], [69.0], [80.0], [69.0]]],

                        gaussian_count=
                        [[[14031.0], [6408.0], [14030.0], [6431.0]],[[9561.0], [3563.0], [9612.0], [3552.0]],[[9668.0], [3064.0], [9724.0], [3099.0]],[[10420.0], [3759.0], [10205.0], [3681.0]],[[10328.0], [5554.0], [10343.0], [4940.0]],[[10601.0], [6864.0], [10601.0], [6373.0]],[[10537.0], [8204.0], [10537.0], [7305.0]],[[10720.0], [9571.0], [10720.0], [7774.0]],[[10720.0], [9571.0], [10720.0], [7774.0]],[[10720.0], [9571.0], [10720.0], [7774.0]]],

                        uniform_count=
                        [[[9092.0], [3606.0], [9051.0], [3391.0]],[[6137.0], [2288.0], [6157.0], [2272.0]],[[7704.0], [2667.0], [7600.0], [2687.0]],[[9935.0], [3991.0], [10162.0], [4152.0]],[[12374.0], [6415.0], [12337.0], [6619.0]],[[13806.0], [8164.0], [13806.0], [8230.0]],[[13986.0], [10344.0], [13986.0], [10783.0]],[[18732.0], [13356.0], [18732.0], [13150.0]],[[18732.0], [13356.0], [18732.0], [13150.0]],[[18732.0], [13356.0], [18732.0], [13150.0]]],

                        grid_count=
                        [[[57524.0], [56847.0], [57543.0], [57110.0]],[[30247.0], [32117.0], [30155.0], [32168.0]],[[23401.0], [24380.0], [23427.0], [24419.0]],[[20689.0], [21514.0], [20671.0], [21609.0]],[[19662.0], [20251.0], [19643.0], [20274.0]],[[19254.0], [19953.0], [19254.0], [19855.0]],[[19158.0], [19754.0], [19158.0], [19645.0]],[[18732.0], [19597.0], [18732.0], [19497.0]],[[18732.0], [19597.0], [18732.0], [19497.0]],[[18732.0], [19597.0], [18732.0], [19497.0]]]
                        )

    max_children_nums = [4, 8, 16, 32, 48, 64, 80, 96, 108, 128]
    labels = ["_R*1","_nonref","_R*2","_ref"]
    count = 1
    for key, values in results_time.items():
        values = [[d[0][0] if d[0][0] !=0 else 1,d[1][0],d[2][0] if d[2][0] !=0 else 1,d[3][0]] for d in values]
        plt.subplot(int('24'+str(count)))
        count += 1
        norefs = [(d[0]-d[1])/d[0] for d in values]
        refs = [(d[2]-d[3])/d[2] for d in values]

        plt.plot(max_children_nums, norefs, color='blue', label=key + "_noref")
        plt.plot(max_children_nums, refs, color='red', label=key + "_ref")
        plt.grid(which='major', axis='x', linewidth=0.75, linestyle='-', color='0.75', dashes=(15, 10))
        plt.grid(which='major', axis='y', linewidth=0.75, linestyle='-', color='0.75', dashes=(15, 10))
        plt.xlabel('max number of child nodes')
        plt.ylabel('Time(s)')
        plt.legend(loc='best')


    for key, values in results_count.items():
        values = [[d[0][0] if d[0][0] != 0 else 1, d[1][0], d[2][0] if d[2][0] != 0 else 1, d[3][0]] for d in values]
        plt.subplot(int('24'+str(count)))
        count += 1
        norefs = [(d[0] - d[1])/d[0] for d in values]
        refs = [(d[2] - d[3])/d[2] for d in values]

        plt.plot(max_children_nums, norefs, color='blue', label=key + "_noref")
        plt.plot(max_children_nums, refs, color='red', label=key + "_ref")
        plt.grid(which='major', axis='x', linewidth=0.75, linestyle='-', color='0.75', dashes=(15, 10))
        plt.grid(which='major', axis='y', linewidth=0.75, linestyle='-', color='0.75', dashes=(15, 10))
        plt.xlabel('max number of child nodes')
        plt.ylabel('number of nodes')
        plt.legend(loc='best')
    plt.show()






if __name__ == '__main__':
    #experiment3()
    drawfig()