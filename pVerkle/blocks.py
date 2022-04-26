
import hashlib
import json
import math
from random import randint

from node import RNode, Entry
from rtree import RTree
from verkle import VerklePatriciaTree
import geo

class BTransaction(Entry):
    def __init__(self,acocunt,log,lat,ts,datas):
        """
        交易信息
        :param account 账户Id
        :param log 经度
        :param lat 维度
        :param ts  交易时间
        :param datas 交易数据
        """
        self.acocunt = acocunt
        self.id = acocunt
        self.log = log
        self.lat = lat
        self.ts = ts
        self.datas = datas
        self.mbr = geo.Rectangle(dimension=3,values=[self.log,self.log,self.lat,self.lat,self.ts,self.ts])

class BRecord(BTransaction):
    def __init__(self,acocunt,log,lat,ts,account_dis):
        """
        轨迹信息
        :account str 账户
        :log 经度
        :lat 纬度
        :ts 时间戳
        :account_dis 账户值
        """
        self.account = acocunt
        self.id = acocunt
        self.log = log
        self.lat = lat
        self.ts = ts
        self.account_dis = account_dis

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


class Block:
    def __init__(self,context,parent_hash,transactions,proof):
        """
        区块
        :param parent_hash
        :param transactions
        :param proof
        """
        self.nonce = proof
        self.parent_hash = parent_hash
        self.next_hash = None
        self.start_time = min((x.ts for x in transactions))
        self.end_time = max((x.ts for x in transactions))
        self.timestamp = self.end_time
        self.statetrieRoot = self._create_vpt(transactions)
        self.trantrieRoot = self._create_tran_trie(transactions)
        self.trajetrieRoot = self._create_traj_trie(transactions)

    def _create_vpt(self,transactions):
        accounts = self._create_accounts(transactions)
        vpt = VerklePatriciaTree()
        for acc in accounts:
            vpt.insert(acc.id,acc)
        self.statetrieRoot = vpt
        return vpt


    def _create_tran_trie(self,transactions):
        mbr = geo.Rectangle.unions([tx.mbr for tx in transactions])
        tree = RTree()
        for tr in transactions:
            tree.insert(tr)
        return tree

    def _create_traj_trie(self,transactions):
        tree = RTree()
        for tr in transactions:
            tree.insert(BRecord(tr.account,tr.log,tr.lat,tr.ts))

    def _create_accounts(self,transactions):
        accounts = {}
        for tx in transactions:
            if tx.account not in accounts:
                accounts[tx.account] = BAccount(tx.account,tx.ts,tx.log,tx.lat)
            elif accounts[tx.account].ts < tx.ts:
                accounts[tx.account] = BAccount(tx.account,tx.ts,tx.log,tx.lat)
        return list(accounts.values())





class BlockChain:
    def __init__(self,context,blocksize=20,transactions=[]):
        self.context = context
        self.blocksize = blocksize
        self.blocks = {}
        self.header = None
        self._create_blocks(transactions)


    @staticmethod
    def hash(block):
        """
        Creates a SHA-256 hash of a Block
        :param block: <dict> Block
        :return: <str>
        """
        # We must make sure that the Dictionary is Ordered, or we'll have inconsistent hashes
        block_string = json.dumps(block, sort_keys=True).encode()
        return hashlib.sha256(block_string).hexdigest()

    def _create_blocks(self,transactions):
        if transactions is None or len(transactions)<=0:return

        block_num = math.ceil(len(transactions)/self.blocksize)

        parent_hash = None
        for i in range(block_num):
            tx = transactions[i*self.blocksize: (i+1)*self.blocksize+1 if (i+1)*self.blocksize<len(transactions) else -1]
            block = Block(self.context,parent_hash,tx,proof = randint(1, 1000))
            hash = BlockChain.hash(block)
            self.blocks[hash] = block
            if parent_hash is None:
                self.header = block
                parent_hash = hash
            else:
                parent = self.blocks[parent_hash]
                parent.next_hash = hash


