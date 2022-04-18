
import utils

class Node:
    def __init__(self):
        self.parent = None
        self.parentKey = -1
class LeafNode(Node):
    def __init__(self,keys,values):
        self.keys = keys
        self.values = values
    def update(self,values):
        self.values = values

class InternalNode(Node):
    pass
class ExtensionNode(InternalNode):
    def __init__(self, nibbs,next=None,value=None):
        self.nibbs = nibbs
        self.next = next
        self.value = value
        if self.value is not None:
            self.value.parent = self
class BranchNode(InternalNode):
    def __init__(self):
        self.entries = []*256
    def __getitem__(self, key):
        return self.entries[key[0]]
    def __setitem__(self, key, node):
        self.entries[key] = node
        node.parent = self
        node.parentKey = key

class VerklePatriciaTree:
    def __init__(self):
        self.root = None

    def insert(self,keys,values,node=None):
        '''
        插入操作  插入的信息包括keys和values两个部分，插入后存储在树的叶子节点中
        @param keys  byte list
        @param values byte list
        '''
        if self.root is None:
            self.root = LeafNode(keys,values)
            return
        if node is LeafNode:
            prefix = utils.maxPrefix(node.keys,keys)
            if node.keys == keys:     #keys与node.keys完全相同
                node.update(values)
            elif len(prefix)==0:      #keys与node.keys完全不同
                bNode = BranchNode()
                bNode[keys[0]]=LeafNode(keys[1:],values)
                bNode[node.keys[0]]=node
                node.keys = node.keys[1:]
                self._modifyParent(node,bNode)
            else:                   #keys与node.keys有相同的前缀
                bNode = BranchNode()
                eNode = ExtensionNode(prefix,bNode)
                eNode.next,bNode.parent = bNode,eNode
                self._modifyParent(node,eNode)
                if prefix == keys:
                    eNode.value = LeafNode(keys[len(prefix):],values)
                    bNode[node.keys[len(prefix)]] = node
                    node.key = node.key[len(prefix)+1:]
                elif prefix == node.keys:
                    eNode.value = node
                    node.keys = node.keys[len(prefix):]
                    bNode[keys[len(prefix)]] = LeafNode(keys[len(prefix)+1:],values)
                else:
                    bNode[keys[len(prefix)]] = LeafNode(keys[len(prefix)+1:], values)
                    bNode[node.keys[len(prefix)]] = node
                    node.key = node.key[len(prefix) + 1:]
        elif node is ExtensionNode:
            if node.nibbs == keys:
                if node.value is Node:
                    node.value = LeafNode(keys,values)
                    node.value.parent = node
                else:
                    node.value.update(values)
            elif node.nibbs.startsWith(keys):
                eNode = ExtensionNode(keys,node,LeafNode(keys,values))
                node.nibbs = node.nibbs[len(keys):]
            elif keys.startsWith(node.nibbs):
                keys = keys[len(node.nibbs):]
                self.insert(keys,values,node.next)
            else:
                bNode = BranchNode()
                self._modifyParent(node,bNode)
                bNode[keys[0]] = LeafNode(keys[1:],values)
                bNode[node.nibbs[0]] = node
                node.nibbs = node.nibbs[1:]
        elif node is BranchNode:
            if node[keys[0]] is None:
                node[keys[0]] = LeafNode(keys[1:],values)
            else:
                keys = keys[1:]
                self.insert(keys,values,node[keys[0]])

    def find(self,keys,node=None):
        '''
        寻找指定keys的叶子节点
        '''
        if node is None:
            node = self.root
        if node is None: return None
        if node is LeafNode:
            return node if node.keys == keys else None
        elif node is ExtensionNode:
            if node.value is not None and node.value.keys == keys:
                return node.value
            else:
                return self.find(keys[len(node.nibbs):],node.next)
        elif node is BranchNode:
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
        if nodeOrigin.parent is BranchNode:
            nodeOrigin.parent[nodeOrigin.parentKey]=nodeNew
        elif nodeOrigin.parent is ExtensionNode:
            nodeOrigin.parent.next = nodeNew
            nodeNew.parent = nodeOrigin.parent
            nodeNew.parentKey = -1


from graphviz import Digraph, nohtml

class VPTView:
    def __init__(self,vpt):
        self.g = None
        self.create(vpt)

    def show(self):
        if self.g is None:
            return
        self.g.view()


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
            self.g.node(gid,utils.bytes2str(node.keys))
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



