
import hashlib
import json
import math
from random import randint

from rtree import RTree
from verkle import VerklePatriciaTree
import geo

class BTransaction:
    def __init__(self,acocunt,log,lat,ts,datas):
        self.account = acocunt
        self.log = log
        self.lat = lat
        self.ts = ts
        self.datas = datas
        self.mbr = geo.Rectangle(dimension=3,values=[self.log,self.log,self.lat,self.lat,self.ts,self.ts])
        self.account_dis = 0
class BRecord(BTransaction):
    ACC_LENGTH = 32
    def __init__(self,acocunt,log,lat,ts):
        self.account = acocunt
        self.log = log
        self.lat = lat
        self.ts = ts

class BAccount:
    def __init__(self,id,ts,log,lat):
        self.id = id
        self.log = log
        self.lat = lat
        self.ts = ts


class Block:
    def __init__(self,context,parent_hash,transactions,proof,range):
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
        self.statetrieRoot,self.trantrieRoot,self.trajetrieRoot = None,None,None

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

    def _create_accounts(self,transactions):
        accounts = {}
        for tx in transactions:
            if tx.account not in accounts:
                accounts[tx.account] = BAccount(tx.account,tx.ts,tx.log,tx.lat)
            elif accounts[tx.tx.account].ts < tx.ts:
                accounts[tx.tx.account] = BAccount(tx.account,tx.ts,tx.log,tx.lat)
        return list(accounts.values())





class BlockChain:
    def __init__(self,context,blocksize=20,transactions=[]):
        self.context = context
        self.blocksize = blocksize
        self._create_blocks(transactions)
        self.blocks = {}
        self.header = None

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


