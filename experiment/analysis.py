import time
from random import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.mixture
from sklearn.mixture import GaussianMixture

from blocks import BTransaction, BlockChain
from dataloader import PocemonLoader
from utils import Configuration


def run_query_transaction():
    # 配置
    context = Configuration({})

    # 读取数据
    dataLoader = PocemonLoader()
    df = dataLoader.refresh()
    df = dataLoader.extend_df(repeat_times=1)
    df = dataLoader.normalize(df)
    transactions = [BTransaction(row['type'],row['log',row['lat'],row['ts'],None]) for row in df.rows]

    # 创建区块链
    chain  = BlockChain(context,blocksize=40,transactions=transactions)

    # 创建历史查询分布
    x = np.arange(0, 2, 1)
    y = np.arange(0, 2, 1)
    z = np.arrage(0, 2, 1)
    meas = np.meshgrid(x,y,z)
    gmm = GaussianMixture(n_components=8, covariance_type='full', random_state=0,means_init=meas)
    gmm.fit(meas)
    # 根据历史查询分布采用查询
    querynum = 1
    qcenters = gmm.sample(querynum)
    qlengths = [(1-np.random())/2 for i in range(querynum)]

    #执行查询
    spans = []
    for i in range(querynum):
        st = time.time()
        chain.query_transations(qcenters[i],qlengths[i])
        spans.append(time.time()-st)
    print(np.average(spans))
