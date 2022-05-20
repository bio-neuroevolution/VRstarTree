
import hashlib
import json
import math
import numpy as np
import os
from random import randint
import  time

from node import RNode, Entry
from rtree import RTree
from utils import Configuration, Collections
from verkle import VerklePatriciaTree, VerkleRTree, VPTView
import geo

class BTransaction(Entry):
    def __init__(self,account,log,lat,ts,datas,geotype=0,length=0):
        """
        交易信息
        :param account 账户Id
        :param log 经度
        :param lat 维度
        :param ts  交易时间
        :param geotype 几何类型
        :param length 几何长度
        :param datas 交易数据
        """
        self.account = account
        self.id = account
        self.log = log
        self.lat = lat
        self.ts = ts
        self.geotype = geotype
        self.length = length
        self.datas = datas
        self._update_mbr()
        self.ref = 0
    def update_geotype(self,geotype,length):
        self.geotype = geotype
        self.length = length
        self._update_mbr()

    def _update_mbr(self):
        if self.geotype == 0: #point
            self.mbr = geo.Rectangle(dimension=3,values=[self.log,self.log,self.lat,self.lat,self.ts,self.ts])
        elif self.geotype == 1:#linesegment
            self.mbr = geo.Rectangle(dimension=3,values=[self.log,self.log + self.length,self.lat,self.lat+self.length,self.ts,self.ts])
        else: # rectangle
            self.mbr = geo.Rectangle(dimension=3,values=[self.log, self.log + self.length/2, self.lat, self.lat + self.length/2,
                                             self.ts, self.ts])
    def __str__(self):
        return self.id+ ',' + str(self.datas) +','+str(self.mbr)+","+str(self.ref)

    def __getitem__(self, item):
        if item == 'ts':return self.ts
        elif item == 'lon': return self.log
        elif item == 'lat': return self.lat
        elif item == 'account': return self.account

class BRecord(BTransaction):
    def __init__(self,account,log,lat,ts,account_dis,geotype=0,length=0.):
        """
        轨迹信息
        :account str 账户
        :log 经度
        :lat 纬度
        :ts 时间戳
        :account_dis 账户值
        """
        super(BRecord,self).__init__(account,log,lat,ts,None,geotype,length)
        self.account_dis = account_dis
        self.ref = 0
        self.mbr = geo.Rectangle(dimension=4,values=[self.log, self.log + self.length/2, self.lat, self.lat + self.length/2,
                                             self.ts, self.ts,self.account_dis,self.account_dis])
    def __str__(self):
        return self.id+ ',' +str(self.mbr)+","+str(self.ref)

    def equals(self,r):
        if self.account != r.account:return False
        if self.log != r.log:return False
        if self.lat != r.lat:return False
        if self.ts != r.ts:return False
        return True

class BAccount:
    def __init__(self,id,ts,log,lat):
        """
        账户信息
        :param id 账户I
        :param log 经度
        :param lat 纬度
        :param ts  时间戳
        """
        self.id = id
        self.log = log
        self.lat = lat
        self.ts = ts
        self.mbr = geo.Rectangle(dimension=3, values=[self.log, self.log, self.lat, self.lat, self.ts, self.ts])
        self.ref = 0
    def __str__(self):
        return self.id+ ',' +str(self.mbr)

class Block:
    def __init__(self,context,parent_hash,transactions,proof):
        """
        区块
        :param parent_hash
        :param transactions
        :param proof
        """
        self.context = context
        self.nonce = proof
        self.parent_hash = parent_hash
        self.next_hash = None
        self.start_time = min((x.ts for x in transactions))
        self.end_time = max((x.ts for x in transactions))
        self.timestamp = self.end_time
        self.statetrieRoot = self._create_vpt(transactions)
        self.trantrieRoot = self._create_tran_trie(transactions)
        self.trajetrieRoot = self._create_traj_trie(transactions)
    def serialize(self):
        rs = {'context': self.context}
        rs.update({'nonce':self.nonce,'parent_hash': str(self.parent_hash),'next_hash':str(self.next_hash)})
        rs.update({'start_time':str(self.start_time),'end_time':str(self.end_time),'timestamp':str(self.timestamp)})
        #rs.update({'statetrieRoot':self.statetrieRoot.serialize()})
        rs.update({'trantrieRoot': self.trantrieRoot.serialize()})
        #rs.update({'trajetrieRoot': self.trajetrieRoot.serialize()})
        return rs

    def summary(self,filename,path):
        print('begin:'+str(self.start_time))
        print('end:'+str(self.end_time))
        vptView = VPTView(self.statetrieRoot)
        vptView.show(filename,path)

    def _create_vpt(self,transactions):
        accounts = self._create_accounts(transactions)
        vpt = None # VerklePatriciaTree()
        #for acc in accounts:
        #    vpt.insert(acc.id,acc)
        self.statetrieRoot = vpt
        return vpt


    def _create_tran_trie(self,transactions):
        mbr = geo.Rectangle.unions([tx.mbr for tx in transactions])
        tree = VerkleRTree(self.context,mbr)
        for tr in transactions:
            tree.insert(tr)
        return tree

    def _create_traj_trie(self,transactions):
        mbr = geo.Rectangle(4,[0.,1.,0.,1.,0.,1.,0.,1.])
        tree = VerkleRTree(self.context,mbr)
        #for tr in transactions:
        #    tree.insert(BRecord(tr.account,tr.log,tr.lat,tr.ts,self.account_distance(tr.account)))
        return tree

    def account_distance(self,s1):
        s2 = "".join(['a']*self.context.account_length)
        b1, b2 = bytearray(s1, encoding='utf-8'), bytearray(s2, encoding='utf-8')
        diff = 0
        for i in range(len(b1)):
            if b1[i] != b2[i]:
                diff += bin(b1[i] ^ b2[i]).count("1")
        return diff/64

    def _create_accounts(self,transactions):
        accounts = {}
        for tx in transactions:
            if tx.account not in accounts:
                accounts[tx.account] = BAccount(tx.account,tx.ts,tx.log,tx.lat)
            elif accounts[tx.account].ts < tx.ts:
                accounts[tx.account] = BAccount(tx.account,tx.ts,tx.log,tx.lat)
        return list(accounts.values())

    def query_account(self, account)->BAccount:
        node = self.statetrieRoot.find(account)
        return None if node is None else node.values

    def query_tran(self,mbr)->list:
        if self.trantrieRoot is None: return []
        if not self.trantrieRoot.range.isOverlop(mbr):return []
        return self.trantrieRoot.find(mbr)
    def query_traj(self,account:str,mbr:geo.Rectangle)->list:
        #r = geo.Rectangle(mbr.dimension + 1)
        #for i in range(mbr.dimension):
        #    r.update(i,mbr.lower(i),mbr.upper(i))
        #r.update(r.dimension-1,account_dis,account_dis)
        account_dis = self.account_distance(account)
        mbr.update(3,account_dis,account_dis)
        return self.trajetrieRoot.find(mbr)





class BlockEnocder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Block):
            return obj.serialize()
        elif isinstance(obj,Configuration):
            return obj.serialize()

        return json.JSONEncoder.default(self, obj)


class BlockChain:
    def __init__(self,context,blocksize=20,transactions=[]):
        self.context = context
        self.blocksize = blocksize
        self.blocks = {}
        self.header = None
        self._create_blocks(transactions)
        self.query_tran_node_count = 0


    @staticmethod
    def hash(block):
        """
        Creates a SHA-256 hash of a Block
        :param block: <dict> Block
        :return: <str>
        """
        # We must make sure that the Dictionary is Ordered, or we'll have inconsistent hashes
        block_string = json.dumps(block, cls=BlockEnocder, sort_keys=True).encode()
        return hashlib.sha256(block_string).hexdigest()

    def _create_blocks(self,transactions):
        if transactions is None or len(transactions)<=0:return

        block_num = math.ceil(len(transactions)/self.blocksize)

        parent_hash = None
        for i in range(block_num):
            tx = transactions[i*self.blocksize: (i+1)*self.blocksize]
            block = Block(self.context,parent_hash,tx,proof = randint(1, 1000))
            hash = BlockChain.hash(block)
            self.blocks[hash] = block
            if parent_hash is None:
                self.header = block
                parent_hash = hash
            else:
                parent = self.blocks[parent_hash]
                parent.next_hash = hash
    def getBlockCount(self):
        return len(self.blocks)

    def next(self,parent=None):
        if parent is None:return self.header
        hash = self.hash(parent)

    def proof_length(self,count,unit):
        return self.header.statetrieRoot.proof_length(count,unit), \
               self.header.trantrieRoot.proof_length(count,unit),  \
               self.header.trajetrieRoot.proof_length(count,unit)

    def summary(self):
        print("context:"+str(self.context.serialize()))
        print("blocksize:"+str(self.blocksize))
        print("block count:"+str(self.getBlockCount()))
        for i,b in self.blocks.items():
            print(str(b.serialize()))
            b.summary('block'+str(i),os.path.abspath('.'))

    def query_account(self,account):
        t,r = None,None
        for i, b in self.blocks.items():
            acc = b.query_account(account)
            if t is None or acc.ts > t:
                t = acc.ts
                r = acc
        return r
    def tran_nodecount(self):
        """
        交易Verkel R*
        -tree树的节点数
        """
        r = 0
        for i, b in self.blocks.items():
            r += b.trantrieRoot.count()
        return r

    def query_tran(self,mbr):
        r,self.query_tran_node_count  = [],0
        for i, b in self.blocks.items():
            txs = b.query_tran(mbr)
            r = r +txs
            self.query_tran_node_count += b.trantrieRoot.query_node_count
        return r
    def query_traj(self,account,mbr):
        r = []
        for i, b in self.blocks.items():
            trajs = b.query_traj(account,mbr)
            r = r + trajs
        rm = []
        for i,t in enumerate(r):
            if Collections.any(lambda x:t.equals(x),rm):
                continue
            rm.append(t)
        return rm

    @classmethod
    def create_query(cls,count=1,sizes:list=[2,2,2],posrandom=100,lengthcenter=0.05,lengthscale=0.025):
        """
        创建满足一定分布的查询
        :param count  int 生成的查询数
        :param sizes  list 生成的查询的每个维的中心点个数
        :param posrandom int 中心点漂移的百分比
        :param lengthcenter float 查询窗口的宽度均值
        :param lengthscale float 查询窗口的宽度方差
        """
        if lengthscale <= 0:
            # 表示在均匀分布上采样
            r = []
            length = lengthcenter
            for c in range(count):
                center = [np.random.random() for dimension in range(len(sizes))]
                mbr = geo.Rectangle(dimension=len(sizes))
                for j in range(len(sizes)):
                    mbr.update(j, center[j] - length / 2, center[j] + length / 2)
                r.append(mbr)
            return r


        # 在多元高斯分布上采样
        ds = []
        for i in range(len(sizes)):
            interval = 1./(sizes[i]+1)
            d = [(j+1)*interval for j in range(sizes[i])]
            ds.append(d)


        r = []
        for c in range(count):
            center = [np.random.choice(d,1)[0] for i,d in enumerate(ds)]
            for i in range(len(center)):
                center[i] += (np.random.random() * 2 - 1) / posrandom
            length = np.random.normal(loc=lengthcenter,scale=lengthscale,size=1)
            mbr = geo.Rectangle(dimension=len(sizes))
            for j in range(len(sizes)):
                mbr.update(j,center[j]-length/2,center[j]+length/2)
            r.append(mbr)
        return r
    def optima(self,refused=True):
        for i, b in self.blocks.items():
            b.trantrieRoot.rearrage_all(refused)



if __name__ == '__main__':
    # 配置
    context = Configuration(max_children_num=6, max_entries_num=3,account_length=8, account_count=200,select_nodes_func='',merge_nodes_func='',split_node_func='')
    # 创建交易信息[0.15,0.45,0.2,0.65,0.3,0.85,0.1,0.8]
    txs = [BTransaction('abcdefgh',0.25,0.75,0.1,'d1'),BTransaction('abhgtcvb',0.15,0.35,0.2,'d2'),
           BTransaction('fbtrcvpn',0.4, 0.5, 0.3,'d3'),BTransaction('ftbnprqw',0.90,0.90,0.4,'d4'),
           BTransaction('uvpocirt',0.6, 0.1, 0.5,'d5'),BTransaction('fbtrcvpn',0.36,0.69,0.6,'d6'),
           BTransaction('abhgtcvb',0.2, 0.2, 0.7,'d7'),BTransaction('qwertyui',0.4, 0.8, 0.8,'d8')]
    # 创建区块
    chain = BlockChain(context=context,blocksize=8,transactions=txs)

    # 打印区块
    chain.summary()

    # 查询账户信息
    print('查询账户（fbtrcvpn）：')
    acc = chain.query_account('fbtrcvpn')
    print('查询账户结果：'+ str(acc))

    # 查询交易信息
    mbr = geo.Rectangle(dimension=4,values=[0.15,0.45,0.2,0.65,0.3,0.85,0.1,0.8])
    print('查询交易：' + str(mbr))
    trx = chain.query_tran(mbr)
    print('查询交易结果：'+str([str(tx) for tx in trx]))

    # 查询轨迹信息
    mbr = geo.Rectangle(dimension=4, values=[0., 1., 0., 1., 0., 1., 0., 1.])
    trajs = chain.query_traj('fbtrcvpn',mbr)
    print('查询轨迹：fbtrcvpn' + str(mbr))
    print('查询轨迹信息：'+ str([str(tr) for tr in trajs]))

    # 创建交易查询分布
    mbrs = chain.create_query(count=100)
    begin = time.time()
    for mbr in mbrs:
        chain.query_tran(mbr)
    print("交易查询消耗（优化前）:"+str(time.time()-begin))

    # 根据查询优化
    chain.optima()

    # 第二次交易查询
    begin = time.time()
    for mbr in mbrs:
        chain.query_tran(mbr)
    print("交易查询消耗（优化后）:" + str(time.time() - begin))



