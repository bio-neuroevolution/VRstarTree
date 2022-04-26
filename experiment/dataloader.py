
import pandas as pd
import glob
from datetime import datetime, timezone
import time
import numpy as np

def get_ts(y, m, d, h=0, mm=0, s=0):
    return int(datetime(y, m, d, h, mm, s).replace(tzinfo=timezone.utc).timestamp())

class PocemonLoader:
    def __init__(self):
        pass
    def refresh(self):
        hk_poke_json = glob.glob('../data/pocemon_hk/hk-*.json')

        def load_poke_json(pokefile):
            return pd.read_json(pokefile).rename(columns={'a': 'lat', 'i': 'type', 't': 'ts', 'o': 'lon'})

        hk_poke_df = pd.concat([load_poke_json(x) for x in hk_poke_json])
        hk_poke_df = hk_poke_df.sort_values(by=['ts'])[['type', 'lat', 'lon', 'ts']]
        shp = hk_poke_df.shape
        hk_poke_df.dropna(axis=0, how='any', inplace=True)
        print(shp, '-(dropna)>', hk_poke_df.shape)
        hk_poke_df['type'] = hk_poke_df['type'].apply(lambda x: str(int(x))).astype(str)
        hk_poke_df['ts'] = hk_poke_df['ts'].astype(int)
        return hk_poke_df.reset_index(drop=True)

    def extend_df(self,df, repeat_times=10, first_time=get_ts(2018, 1, 1), bs=20):
        a = pd.concat([df] * repeat_times, ignore_index=True)
        MS = 1000
        dt = MS / bs
        new_times = np.arange(first_time * MS, first_time * MS + dt * a.shape[0], dt) / MS
        new_times = new_times[:a.shape[0]]
        new_times = new_times.astype(int)
        a.ts = new_times
        print(df.shape, '-(rept{})>'.format(repeat_times), a.shape)
        return a

    def normalize(self,df):
        max_min_scaler = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
        df[['lat']].apply(max_min_scaler)
        df[['lon']].apply(max_min_scaler)
        df[['ts']].apply(max_min_scaler)
        return df

    def create_account_name(self,count,length)->(list,list):
        """
        创建账户信息
        :param count int 账户数
        :param length int 每个账户位数
        :return (list,list) 账户名集，海明距离集
        """
        symbols = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
        base = ['a']*length
        names = [''.join(np.random.choice(symbols, length)) for i in range(count)]
        dis  = [self.hamming_distance(s,base) for s in names]
        dis = (dis - min(dis))/(max(dis)-min(dis))
        return names,dis

    def hamming_distance(self,s1, s2):
        b1, b2 = bytearray(s1, encoding='utf-8'), bytearray(s2, encoding='utf-8')
        diff = 0
        for i in range(len(b1)):
            if b1[i] != b2[i]:
                diff += bin(b1[i] ^ b2[i]).count("1")
        return diff




