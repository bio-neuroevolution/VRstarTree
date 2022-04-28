import dumb25519
import polycommit
from dumb25519 import Scalar
from polynomial import lagrange
from rtree import RTree


class Node:
    def __init__(self):
        '''
        VerklePatriciaTree的节点
        :param parent Node 父节点
        :param parentKey list of byte 父节点的key
        :param commitment 向量承诺值
        :param poly
        :param scalar
        '''
        self.parent = None
        self.parentKey = -1
        self.commitment = 0
        self.poly = 0
        self.scalar = 0

class LeafNode(Node):
    def __init__(self,keys,values,fullkeys):
        """
        叶节点
        :param keys 叶节点的key
        :param values 叶节点的值
        :param fullkeys 叶子节点的全key
        """
        super(LeafNode,self).__init__()
        self.keys = keys
        self.values = values
        self.fullkeys = fullkeys
    def update(self,values):
        self.values = values

class InternalNode(Node):
    def __init__(self):
        super(InternalNode, self).__init__()

class ExtensionNode(InternalNode):
    def __init__(self, nibbs,next=None,value=None):
        '''
        扩展节点
        :param nibbs list of byte 扩展节点的key
        :param next Node 下一个节点
        :param value 扩展节点关联的值
        '''
        super(ExtensionNode, self).__init__()
        self.nibbs = nibbs
        self.next = next
        self.value = value
        if self.value is not None:
            self.value.parent = self
class BranchNode(InternalNode):
    ENTRY_NUM = 256
    def __init__(self):
        '''
        分支节点
        '''
        super(BranchNode, self).__init__()
        self.entries = [-1]*BranchNode.ENTRY_NUM
    def __getitem__(self, key):
        '''
        访问入口
        '''
        v = self.entries[ord(key[0])]
        return None if v is None or v == -1 else v

    def __setitem__(self, key, node):
        '''
        修改入口
        '''
        self.entries[ord(key)] = node
        if node is not None:
            node.parent = self
            node.parentKey = key

class VerklePatriciaTree:
    def __init__(self):
        self.root = None
        self.vecnum = 255
        self.G = dumb25519.PointVector([dumb25519.random_point() for i in range(self.vecnum)])

    def insert(self,keys,values,node=None,fullkeys = None):
        '''
        插入操作  插入的信息包括keys和values两个部分，插入后存储在树的叶子节点中
        @param keys  byte list
        @param values byte list
        '''
        if fullkeys is None:
            fullkeys = keys
        if self.root is None:
            self.root = LeafNode(keys,values,fullkeys)
            return
        if node is None:
            node = self.root
        if isinstance(node,LeafNode):
            prefix = self.maxPrefix(node.keys,keys)
            if node.keys == keys:     #keys与node.keys完全相同
                node.update(values)
            elif len(prefix)==0:      #keys与node.keys完全不同
                bNode = BranchNode()
                self._modifyParent(node,bNode)
                bNode[keys[0]]=LeafNode(keys[1:],values,fullkeys)
                bNode[node.keys[0]]=node
                node.keys = node.keys[1:]
            else:                   #keys与node.keys有相同的前缀
                bNode = BranchNode()
                eNode = ExtensionNode(prefix,bNode)
                eNode.next,bNode.parent = bNode,eNode
                self._modifyParent(node,eNode)
                if prefix == keys:
                    eNode.value = LeafNode(keys[len(prefix):],values,fullkeys)
                    bNode[node.keys[len(prefix)]] = node
                    node.key = node.key[len(prefix)+1:]
                elif prefix == node.keys:
                    eNode.value = node
                    node.keys = node.keys[len(prefix):]
                    bNode[keys[len(prefix)]] = LeafNode(keys[len(prefix)+1:],values,fullkeys)
                else:
                    bNode[keys[len(prefix)]] = LeafNode(keys[len(prefix)+1:], values,fullkeys)
                    bNode[node.keys[len(prefix)]] = node
                    node.keys = node.keys[len(prefix) + 1:]
        elif isinstance(node,ExtensionNode):
            if node.nibbs == keys:
                if node.value is Node:
                    node.value = LeafNode(keys,values,fullkeys)
                    node.value.parent = node
                else:
                    node.value.update(values)
            elif node.nibbs.startswith(keys):
                eNode = ExtensionNode(keys,node,LeafNode(keys,values,fullkeys))
                node.nibbs = node.nibbs[len(keys):]
            elif keys.startswith(node.nibbs):
                keys = keys[len(node.nibbs):]
                self.insert(keys,values,node.next)
            else:
                bNode = BranchNode()
                self._modifyParent(node,bNode)
                bNode[keys[0]] = LeafNode(keys[1:],values,fullkeys)
                bNode[node.nibbs[0]] = node
                node.nibbs = node.nibbs[1:]
        elif isinstance(node,BranchNode):
            if node[keys[0]] is None:
                node[keys[0]] = LeafNode(keys[1:],values,fullkeys)
            else:
                self.insert(keys=keys[1:],values=values,node=node[keys[0]],fullkeys=fullkeys)

    def maxPrefix(self,text1, text2):
        r = []
        for i in range(min(len(text1), len(text2))):
            if text1[i] == text2[i]:
                r.append(text1[i])
            else:
                break
        return "".join(r)

    def find(self,keys,node=None):
        '''
        寻找指定keys的叶子节点
        '''
        if node is None:
            node = self.root
        if node is None: return None
        if isinstance(node,LeafNode):
            return node if node.keys == keys else None
        elif isinstance(node,ExtensionNode):
            if node.value is not None and node.value.keys == keys:
                return node.value
            else:
                return self.find(keys[len(node.nibbs):],node.next)
        elif isinstance(node,BranchNode):
            if node[keys[0]] is None:
                return None
            return self.find(keys[1:],node[keys[0]])



    def _modifyParent(self,nodeOrigin,nodeNew):
        '''
        将nodeNew的父节点设置为nodeOrigin的父节点
        '''
        if nodeOrigin.parent is None:
            self.root = nodeNew
            return
        if nodeNew is None:return

        if isinstance(nodeOrigin.parent,BranchNode):
            nodeNew.parentKey = nodeOrigin.parentKey
            nodeOrigin.parent[nodeOrigin.parentKey]=nodeNew
            nodeNew.parent = nodeOrigin.parent
        elif isinstance(nodeOrigin.parent, ExtensionNode):
            nodeOrigin.parent.next = nodeNew
            nodeNew.parent = nodeOrigin.parent
            nodeNew.parentKey = nodeOrigin.parentKey

    def _computeCommitment(self,node=None):
        if node is None:
            node = self.root
        if isinstance(node,LeafNode):
            nodehash = [dumb25519.hash_to_scalar('verkle', node)]
            node.poly, node.scalar, node.commitment = self._computePoly(nodehash)
        elif isinstance(node,ExtensionNode):
            if node.next is None:
                node.poly,node.scalar,node.commitment = node.next.poly,node.next.scalar,node.next.commitment
            else:
                self._computeCommitment(node.next)
                nodehash = [dumb25519.hash_to_scalar('verkle', node.next),dumb25519.hash_to_scalar('verkle', node.value)]
                node.poly, node.scalar, node.commitment = self._computePoly(nodehash)
        elif isinstance(node,BranchNode):
            for i in range(node.entries):
                if node.entries[i] is None:
                    continue
                self._computeCommitment(node.entries[i])
            nodehash = [dumb25519.hash_to_scalar('verkle', node) for node in node.entries if node is not None]
            node.poly,node.scalar,node.commitment = self._computePoly(nodehash)

    def _computePoly(self,nodehash):
        coords = [(Scalar(j + 1), hash) for j, hash in enumerate(nodehash)]
        poly = lagrange(coords)  # polynomial coefficients
        scalar = dumb25519.random_scalar()  # blinding factor
        commitment = poly ** self.G + scalar * polycommit.H  # the actual commitment
        return poly,scalar,commitment


class VerkleRTree(RTree):
    def __init__(self,context,range):
        super(VerkleRTree,self).__init__(context,range)

from graphviz import Digraph, nohtml

class VPTView:
    def __init__(self,vpt):
        self.g = None
        self.create(vpt)

    def show(self,filename,directory):
        if self.g is None:
            return
        self.g.view(filename=filename,directory=directory)


    def create(self,vpt,node=None,parent=None,id=1):
        if vpt is None:
            return
        if node is None:
            node = vpt.root
        if node is None:
            return
        if self.g is None:
            self.g = Digraph('g', filename='btree.gv',node_attr={'shape': 'record', 'height': '.1'})
        if node is LeafNode:
            gid = str(id)
            self.g.node(gid,node.keys)
            if parent is not None:
                self.g.edges([(parent,gid)])
        elif node is ExtensionNode:
            gid = str(id)
            gn = self.g.node(gid,nohtml('<f0>'+id+'|<f1> next|<f2>'))
            if parent is not None:
                self.g.edges([(parent,gid)])
            if node.value is not None:
                id += 1
                cid = str(id)
                gc = self.g.node(cid)
                self.g.edges([(gid+':f2',cid)])
            if node.next is None:
                return
            self.create(vpt,node.next,self.g,gid,id+1)
        elif node is BranchNode:
            content = ''
            for i in range(len(node.entries)):
                if node.entries[i] is None:
                    continue
                content = content + '<f'+str(i)+'>'+chr(i)+'|'
            content = content[:-1]
            gid = str(id)
            gn = self.g.node(id,nohtml(content))
            if parent is not None:
                self.g.edges([(parent,gid)])
            for i in range(len(node.entries)):
                if node.entries[i] is None:
                    continue
                id += 1
                self.create(vpt,node.entries[i],self.g,gid+':f'+str(i))




