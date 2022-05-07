
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
        print('数据量:', hk_poke_df.shape)
        hk_poke_df['type'] = hk_poke_df['type'].apply(lambda x: str(int(x))).astype(int)
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
        #print(df.shape, '-(rept{})>'.format(repeat_times), a.shape)
        return a

    def normalize(self,df):
        cols = ['lat','lon','ts']
        for col in cols:
            maxv = df[col].max()
            minv = df[col].min()
            df[col] = (df[col]-minv)/(maxv-minv)
        return df

    def create_region(self,transactions,geotype_probs,length_probs,lengthcenters,lengthscales):
        num = len(transactions)
        geotypes = np.random.choice(a=[0,1,2], size=num, replace=True, p=geotype_probs)
        for i,tr in enumerate(transactions):
            if geotypes[i] == 0:
                tr.update_geotype(0,0)
            else:
                lengthindex = np.random.choice(a=range(len(length_probs)), size=1, replace=True, p=length_probs)[0]
                length = np.random.normal(loc=lengthcenters[lengthindex], scale=lengthscales[lengthindex], size=1)
                tr.update_geotype(geotypes[i], length)



    def create_account_name(self,count,length)->(list,list):
        """
        创建账户信息
        :param count int 账户数
        :param length int 每个账户位数
        :return (list,list) 账户名集，海明距离集
        """
        symbols = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
        base = "".join(['a']*length)
        names = [''.join(np.random.choice(symbols, length)) for i in range(count)]
        dis  = [self.hamming_distance(s,base) for s in names]
        mindis,maxdis = min(dis),max(dis)
        dis = [(d - mindis)/(maxdis-mindis) for d in dis]
        return names,dis

    def hamming_distance(self,s1, s2):
        b1, b2 = bytearray(s1, encoding='utf-8'), bytearray(s2, encoding='utf-8')
        diff = 0
        for i in range(len(b1)):
            if b1[i] != b2[i]:
                diff += bin(b1[i] ^ b2[i]).count("1")
        return diff




