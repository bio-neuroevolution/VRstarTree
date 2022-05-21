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
                        #center = dict(count=6000, sizes=[1,1,1], posrandom=100, lengthcenter=0.1, lengthscale=0.05),
                        gaussian = dict(count=6000, sizes=[2,2,2], posrandom=100, lengthcenter=0.05, lengthscale=0.1)
                        #uniform = dict(count=6000, sizes=[2, 2, 2], posrandom=100, lengthcenter=0.1, lengthscale=0.0),
                        #grid= dict(count=6000, sizes=[2, 2, 2], posrandom=100, lengthcenter=0.05, lengthscale=-1.0)
                        ),
                max_children_nums = [8], #[4,8,16,32,48,64,80,96,108,128],
                blocksizes = [80],
                fig='show',
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
    for key, values in rtree_time.items():
        plt.subplot(int('24' + str(count)))
        count += 1
        norefs = [(d[0][0] - d[1][0]) for d in values]
        refs = [(d[2][0] - d[3][0]) for d in values]

        plt.plot(max_children_nums, norefs, color='blue', label=key + "_noref")
        plt.plot(max_children_nums, refs, color='red', label=key + "_ref")
        plt.grid(which='major', axis='x', linewidth=0.75, linestyle='-', color='0.75', dashes=(15, 10))
        plt.grid(which='major', axis='y', linewidth=0.75, linestyle='-', color='0.75', dashes=(15, 10))
        plt.xlabel('max number of child nodes')
        plt.ylabel('Time(s)')
        plt.legend(loc='best')

    for key, values in rtree_count.items():
        plt.subplot(int('24' + str(count)))
        count += 1
        norefs = [(d[0][0] - d[1][0]) for d in values]

        # max, min = np.max(norefs), np.min(norefs)
        # norefs = [(d - min) / (max - min) for d in norefs]
        refs = [(d[2][0] - d[3][0]) for d in values]

        print(key)
        print(norefs)
        print(refs)
        # max, min = np.max(refs), np.min(refs)
        # refs = [(d - min) / (max - min) for d in refs]
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
                        [[[14.694713592529297], [14.781670093536377], [14.474470376968384], [15.144961833953857]],
                         [[14.184765338897705], [14.772777080535889], [14.129993915557861], [14.636929512023926]],
                         [[14.146383285522461], [14.823366641998291], [14.334093809127808], [15.076643228530884]],
                         [[14.30675458908081], [15.32998538017273], [14.53315258026123], [15.335512638092041]],
                         [[15.282330513000488], [15.72642970085144], [14.76853084564209], [15.185842037200928]],
                         [[14.489277124404907], [15.846860647201538], [14.420629024505615], [15.629188060760498]],
                         [[15.792757987976074], [15.903490781784058], [16.14185357093811], [16.15980577468872]],
                         [[15.579358100891113], [15.621243238449097], [15.917656898498535], [16.09432053565979]],
                         [[15.583758115768433], [15.896770238876343], [15.835871934890747], [16.243523120880127]],
                         [[15.566391468048096], [16.076029062271118], [15.765857696533203], [16.229837894439697]]],
                        gausian_time=
                        [[[21.670360326766968], [17.666773080825806], [22.560691118240356], [17.47241473197937]],
                         [[19.13358163833618], [16.734964609146118], [19.189738750457764], [17.12821674346924]],
                         [[17.982388734817505], [16.40713667869568], [18.813862323760986], [17.465281009674072]],
                         [[18.397822856903076], [18.253204584121704], [19.215627908706665], [18.12953209877014]],
                         [[17.792412996292114], [17.024465799331665], [17.717758893966675], [17.357733488082886]],
                         [[17.457336902618408], [18.088644981384277], [17.48127245903015], [17.689686059951782]],
                         [[18.47760820388794], [18.60748863220215], [18.50899386405945], [18.538323402404785]],
                         [[18.51571822166443], [18.715553760528564], [18.63562273979187], [18.965583562850952]],
                         [[18.17641544342041], [18.676501274108887], [18.400062322616577], [18.55932307243347]],
                         [[18.684390544891357], [18.738882303237915], [18.286121368408203], [18.48782467842102]]],
                        uniform_time=
                        [[[27.70582103729248], [15.428723096847534], [28.527039289474487], [15.671078443527222]],
                         [[21.589289903640747], [12.772908687591553], [21.803715229034424], [12.748915672302246]],
                         [[19.70931100845337], [13.564735174179077], [19.411107063293457], [12.80074167251587]],
                         [[16.608142852783203], [12.997827529907227], [16.702619552612305], [13.090004205703735]],
                         [[17.547131776809692], [14.503260612487793], [17.463319540023804], [14.469313144683838]],
                         [[16.350524425506592], [15.223673343658447], [16.42139768600464], [15.03281283378601]],
                         [[16.556753158569336], [16.1510648727417], [16.659958362579346], [16.510112762451172]],
                         [[16.665460109710693], [16.201693773269653], [16.105950355529785], [16.653485536575317]],
                         [[16.109176635742188], [16.153146982192993], [16.675615787506104], [16.750253915786743]],
                         [[16.232643842697144], [16.09917664527893], [16.40218496322632], [16.607864141464233]]],
                        grid_time=
                        [[[3.6053292751312256], [3.5772576332092285], [3.539799213409424], [3.6631741523742676]],
                         [[3.368988037109375], [3.39092755317688], [3.538742780685425], [3.4417965412139893]],
                         [[3.3909289836883545], [3.3739750385284424], [3.355031967163086], [3.3659961223602295]],
                         [[3.174684762954712], [3.195458173751831], [3.1685307025909424], [3.225625514984131]],
                         [[3.226381301879883], [3.1617913246154785], [3.14459490776062], [3.153566598892212]],
                         [[3.1326277256011963], [3.162510871887207], [3.1635549068450928], [3.2353525161743164]],
                         [[3.158555507659912], [3.098716974258423], [3.123654842376709], [3.2453153133392334]],
                         [[3.2134110927581787], [3.1089203357696533], [3.095724105834961], [3.1039376258850098]],
                         [[3.138582468032837], [3.1046876907348633], [3.118664264678955], [3.117666482925415]],
                         [[3.1445908546447754], [3.227372884750366], [3.1278555393218994], [3.1336164474487305]]],
                        )
    results_count = dict(
                        center_count=
                        [[[3.0], [7.0], [1.0], [7.0]], [[1.0], [8.0], [1.0], [10.0]], [[1.0], [11.0], [1.0], [12.0]],
                         [[1.0], [22.0], [1.0], [28.0]], [[37.0], [48.0], [37.0], [26.0]],
                         [[21.0], [62.0], [21.0], [52.0]], [[80.0], [69.0], [80.0], [69.0]],
                         [[80.0], [69.0], [80.0], [69.0]], [[80.0], [69.0], [80.0], [69.0]],
                         [[80.0], [69.0], [80.0], [69.0]]],

                        gaussian_count=
                        [[[3510.0], [1610.0], [3519.0], [1601.0]], [[3742.0], [1420.0], [3690.0], [1445.0]],
                         [[4228.0], [1818.0], [4295.0], [1827.0]], [[4464.0], [2529.0], [4388.0], [2513.0]],
                         [[4454.0], [2803.0], [4454.0], [2833.0]], [[4636.0], [3681.0], [4636.0], [3670.0]],
                         [[4720.0], [3816.0], [4720.0], [3816.0]], [[4720.0], [3816.0], [4720.0], [3816.0]],
                         [[4720.0], [3816.0], [4720.0], [3816.0]], [[4720.0], [3816.0], [4720.0], [3816.0]]],

                        uniform_count=
                        [[[6490.0], [2731.0], [6336.0], [2755.0]], [[7397.0], [2919.0], [7331.0], [2903.0]],
                         [[9193.0], [3960.0], [9207.0], [3935.0]], [[11771.0], [5802.0], [11894.0], [5795.0]],
                         [[13397.0], [7607.0], [13397.0], [7607.0]], [[14083.0], [9736.0], [14083.0], [9736.0]],
                         [[18732.0], [12797.0], [18732.0], [12797.0]], [[18732.0], [12797.0], [18732.0], [12797.0]],
                         [[18732.0], [12797.0], [18732.0], [12797.0]], [[18732.0], [12797.0], [18732.0], [12797.0]]],

                        grid_count=
                        [[[30183.0], [31940.0], [30152.0], [32056.0]], [[23435.0], [24348.0], [23418.0], [24434.0]],
                         [[20677.0], [21545.0], [20688.0], [21626.0]], [[19641.0], [20282.0], [19652.0], [20311.0]],
                         [[19254.0], [19834.0], [19254.0], [19843.0]], [[19158.0], [19649.0], [19158.0], [19656.0]],
                         [[18732.0], [19497.0], [18732.0], [19497.0]], [[18732.0], [19497.0], [18732.0], [19497.0]],
                         [[18732.0], [19497.0], [18732.0], [19497.0]], [[18732.0], [19497.0], [18732.0], [19497.0]]]
                        )

    max_children_nums = [4, 8, 16, 32, 48, 64, 80, 96, 108, 128]
    labels = ["_R*1","_nonref","_R*2","_ref"]
    count = 1
    for key, values in results_time.items():
        plt.subplot(int('24'+str(count)))
        count += 1
        norefs = [(d[0][0]-d[1][0]) for d in values]
        refs = [(d[2][0]-d[3][0]) for d in values]

        plt.plot(max_children_nums, norefs, color='blue', label=key + "_noref")
        plt.plot(max_children_nums, refs, color='red', label=key + "_ref")
        plt.grid(which='major', axis='x', linewidth=0.75, linestyle='-', color='0.75', dashes=(15, 10))
        plt.grid(which='major', axis='y', linewidth=0.75, linestyle='-', color='0.75', dashes=(15, 10))
        plt.xlabel('max number of child nodes')
        plt.ylabel('Time(s)')
        plt.legend(loc='best')


    for key, values in results_count.items():

        plt.subplot(int('24'+str(count)))
        count += 1
        norefs = [(d[0][0] - d[1][0]) for d in values]

        #max, min = np.max(norefs), np.min(norefs)
        #norefs = [(d - min) / (max - min) for d in norefs]
        refs = [(d[2][0] - d[3][0]) for d in values]

        print(key)
        print(norefs)
        print(refs)
        #max, min = np.max(refs), np.min(refs)
        #refs = [(d - min) / (max - min) for d in refs]
        plt.plot(max_children_nums, norefs, color='blue', label=key + "_noref")
        plt.plot(max_children_nums, refs, color='red', label=key + "_ref")
        plt.grid(which='major', axis='x', linewidth=0.75, linestyle='-', color='0.75', dashes=(15, 10))
        plt.grid(which='major', axis='y', linewidth=0.75, linestyle='-', color='0.75', dashes=(15, 10))
        plt.xlabel('max number of child nodes')
        plt.ylabel('number of nodes')
        plt.legend(loc='best')
    plt.show()






if __name__ == '__main__':
    experiment3()
    #drawfig()