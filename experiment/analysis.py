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
    context = Configuration(max_children_num=16,account_length=16,account_count=200)

    # 读取数据
    dataLoader = PocemonLoader()
    df = dataLoader.refresh()
    df = dataLoader.extend_df(df=df,repeat_times=1)
    #df = (df - df.min()) / (df.max() - df.min())
    df = dataLoader.normalize(df)
    transactions = [BTransaction(row['type'],row['lon'],row['lat'],row['ts'],None) for index,row in df.iterrows()]

    #创建和分配账户
    account_name,account_dis = dataLoader.create_account_name(context.account_count,context.account_length)
    accs = np.random.choice(account_name,len(transactions))
    diss = [account_dis[account_name.index(a)] for a in accs]
    for i,tr in enumerate(transactions):
        tr.account = accs[i]
        tr.account_dis = diss[i]


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

if __name__ == '__main__':
    run_query_transaction()