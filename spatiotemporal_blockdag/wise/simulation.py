#!/usr/bin/python
# encoding: utf-8

import pandas as pd
import glob
import matplotlib.pyplot as plt
import numpy as np
from random import randint
from datetime import datetime

from blockDAG.blockDAG import BlockDAGlocal
from blockDAG.geo_utils import to_Cartesian
from wise.utils import get_ts

def plot_date_hist(df):
    df = df.copy()
    df["date"] = df["ts"].astype("datetime64[s]")
    df[['date', 'type']].groupby([df["date"].dt.year, df["date"].dt.month]).count().plot(kind="bar")

class GeneratorDAGchain:

    datasets = {'hk' : ''}

    def __init__(self, N, tr=60, D=3, bs=40, alpha=10):
        '''
        k_0 (D*lambda)

        '''
        self.bs = bs
        self.N = N
        self.topology = self.generate_topology_dag(N = N, tr=tr, D=D, bs=bs, alpha=alpha)

    # Main method
    @staticmethod
    def generate(repeat_times = 5, tr=60, D=3, bs=40, alpha=10,geotype_probs=[], length_probs=[], lengthcenters=[], lengthscales=[]):
        df = GeneratorDAGchain.read_prepare_dataset('hk')
        df = GeneratorDAGchain.extend_df(df, repeat_times=repeat_times, bs=bs)
        #print([datetime.utcfromtimestamp(df.ts.values[ti]) for ti in (0,-1)])
        g = GeneratorDAGchain(df.shape[0], tr=tr, D=D, bs=bs, alpha=alpha)

        #print('block size', bs)
        #s = sum(x * bs for x in g.topology)
        #print('Max capacity for transactions', s, s - df.shape[0])

        #plot_date_hist(df)

        GeneratorDAGchain._extends_region(df,geotype_probs,length_probs,lengthcenters,lengthscales)

        block_dag = g.generate_blockDAG(df,len(geotype_probs)<=0)
        block_dag.min_lat = df['lat'].min()
        block_dag.max_lat = df['lat'].max()
        block_dag.min_lon = df['lon'].min()
        block_dag.max_lon = df['lon'].max()
        block_dag.min_ts = df['ts'].min()
        block_dag.max_ts = df['ts'].max()
        return block_dag

    @staticmethod
    def _size(df):
        """
        计算
        """
        min_lat = df['lat'].min()
        max_lat = df['lat'].max()
        min_lon = df['lon'].min()
        max_lon = df['lon'].max()
        min_ts = df['ts'].min()
        max_ts = df['ts'].max()

        x1,y1 = to_Cartesian(min_lat,min_lon)
        x2,y2 = to_Cartesian(max_lat,max_lon)
        return abs(x1-x2),abs(y1-y2),max_ts - min_ts

    @staticmethod
    def _extends_region(df,geotype_probs,length_probs,lengthcenters,lengthscales):
        num = df.shape[0]

        w,h,t = GeneratorDAGchain._size(df)
        ran = min(w,h)

        df_geotype,df_length = np.zeros(num),np.zeros(num)

        if geotype_probs is not None and len(geotype_probs)>0:
            geotypes = np.random.choice(a=[0,1,2], size=num, replace=True, p=geotype_probs)

        for i, row in df.iterrows():
            if len(geotype_probs) == 0:
                df_geotype[i],df_length[i] = 0,0.
                continue

            df_geotype[i] = geotypes[i]
            if geotypes[i] == 0:
                df_length[i] = 0
            else:
                lengthindex = np.random.choice(a=range(len(length_probs)), size=1, replace=True, p=length_probs)[0]
                length = np.random.normal(loc=lengthcenters[lengthindex], scale=lengthscales[lengthindex], size=1)
                if length < 0 : length = lengthcenters[lengthindex]
                df_length[i] = length*ran

        df.insert(3, 'geotype', df_geotype)
        df.insert(4, 'length', df_length)

    def __gen_transaction_sizes_per_sec(self, tr, alpha, N):
        """
        tr: average transaction rate per second
        alpha: 3 sigma from average (consider as bounds)
        N: total number of transactions
        return list of transactions per second
        """
        mu, sigma = tr, alpha // 3  # mean and standard deviation
        maximum_number_blocks = N // (tr - alpha)
        #print('maximum_number_blocks', maximum_number_blocks)
        block_sizes = np.random.normal(mu, sigma, maximum_number_blocks).astype(int)

        #count, bins, ignored = plt.hist(block_sizes, 30, density=True)
        #plt.plot(bins, 1 / (sigma * np.sqrt(2 * np.pi)) *
        #         np.exp(- (bins - mu) ** 2 / (2 * sigma ** 2)),
        #         linewidth=2, color='r')

        resulting_block_sizes = block_sizes
        tr_sum = sum(block_sizes)
        for i in range(1, len(block_sizes) + 1):
            if tr_sum - block_sizes[-i] < N:
                resulting_block_sizes = np.append(block_sizes[:-i], N - sum(block_sizes[:-i]))
                break
            elif tr_sum - block_sizes[-i] == N:
                resulting_block_sizes = block_sizes[:-i]
                break
            else:
                tr_sum -= block_sizes[-i]

        #print('len of resulting_block_sizes', len(resulting_block_sizes))
        assert sum(resulting_block_sizes) == N
        return resulting_block_sizes


    def __gen_transaction_numbers_per_delay(self, tr_rates, D):
        trs = tr_rates.copy()
        num_tail = len(trs) % D
        if num_tail:
            trs = np.append(trs, [0] * (D - num_tail))

        trs = trs.reshape((len(trs) // D, D))
        #print(trs.shape)
        assert np.sum(trs) == self.N
        return trs


    def __generate_blocks_number_per_delay(self, tr_rates_delay, bs):
        prev = 0
        block_numbers = []
        for x in tr_rates_delay:
            s = sum(x) + prev
            block_numbers.append(s // bs)
            prev = s % bs
        if prev:
            import math
            block_numbers[-1] += math.ceil(prev / bs)
        return block_numbers


    def generate_topology_dag(self, N, tr=60, D=3, bs=40, alpha=10):
        """
        - block size (bs)
        - average transaction arrival rate (tr) transactions per second (-alpha, tr, alpha)
        - propagation delay (D) in seconds average between a pair of nodes in a peer network
        - alpha: 3 sigma from average transaction arrival rate
        """
        tr_rates = self.__gen_transaction_sizes_per_sec(tr, alpha, N)
        tr_rates_delay = self.__gen_transaction_numbers_per_delay(tr_rates, D)
        blocks_per_delay = self.__generate_blocks_number_per_delay(tr_rates_delay, bs)
        return blocks_per_delay

    @staticmethod
    def read_prepare_dataset(dataset_key='hk'):
        if dataset_key != 'hk':
            exit(1)
        def load_poke_json(pokefile):
            return pd.read_json(pokefile).rename(columns={'a': 'lat', 'i': 'type', 't': 'ts', 'o': 'lon'})

        hk_poke_json = glob.glob('../../data/pocemon_hk/hk-*.json')
        assert hk_poke_json
        hk_poke_df = pd.concat([load_poke_json(x) for x in hk_poke_json])
        hk_poke_df = hk_poke_df.sort_values(by=['ts'])[['type', 'lat', 'lon', 'ts']]
        shp = hk_poke_df.shape
        hk_poke_df.dropna(axis=0, how='any', inplace=True)
        #print(shp, '-(dropna)>', hk_poke_df.shape)
        hk_poke_df['type'] = hk_poke_df['type'].apply(lambda x: str(int(x))).astype(str)
        hk_poke_df['ts'] = hk_poke_df['ts'].astype(int)
        return hk_poke_df.reset_index(drop=True)

    @staticmethod
    def extend_df(df, repeat_times=10, first_time=get_ts(2018, 1, 1), bs = 20):
        a = pd.concat([df] * repeat_times, ignore_index=True)
        MS = 1000
        dt = MS / bs
        new_times = np.arange(first_time * MS, first_time * MS + dt * a.shape[0], dt) / MS
        new_times = new_times[:a.shape[0]]
        new_times = new_times.astype(int)
        a.ts = new_times
        #print(df.shape, '-(rept{})>'.format(repeat_times), a.shape)
        return a



    def generate_blockDAG(self, df,pointonly=True):
        topology = self.topology
        bs = self.bs

        def chunker(seq, size):
            return (seq[pos:pos + size] for pos in range(0, len(seq), size))

        block_dag = BlockDAGlocal()
        gen_transactions_per_block = chunker(df, bs)
        for block_count in topology:
            for i in range(block_count):
                trxs = next(gen_transactions_per_block)
                if trxs is not None and len(trxs) > 0:
                    for type, lon, lat, ts,geotype,length in trxs[['type', 'lon', 'lat', 'ts','geotype','length']].values:
                        block_dag.new_transaction(account=type, lng=lon, lat=lat, ts=ts,geotype=geotype,length=length)
                    # pseudo PoW
                    block_dag.create_block(proof = randint(1, 1000),pointonly=pointonly)
                else:
                    print('Strange:', trxs)
            block_dag.update_orphan_hashes()
        block_dag.optimize()
        return block_dag


if __name__ == '__main__':
    pass