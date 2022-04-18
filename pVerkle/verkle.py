
import utils

class Node:
    def __init__(self):
        self.parent = None
        self.parentKey = -1
class LeftNode(Node):
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
class BranchNode(InternalNode):
    def __init__(self):
        self.entries = []*256
    def __getitem__(self, key):
        return self.entries[key[0]]
    def __setitem__(self, key, node):
        self.entries[key] = node
        node.parent = self
        node.parentKey = key

class VerkleTree:
    def __init__(self):
        self.root = None

    def insert(self,keys,values,node=None):
        if self.root is None:
            self.root = LeftNode(keys,values)
            return
        if node is LeftNode:
            prefix = utils.maxPrefix(node.keys,keys)
            if node.keys == keys:     #keys与node.keys完全相同
                node.update(values)
            elif len(prefix)==0:      #keys与node.keys完全不同
                bNode = BranchNode()
                bNode[keys[0]]=LeftNode(keys[1:],values)
                bNode[node.keys[0]]=node
                node.keys = node.keys[1:]
                self._modifyParent(node,bNode)
            else:
                bNode = BranchNode()
                eNode = ExtensionNode(prefix,bNode)
                eNode.next,bNode.parent = bNode,eNode
                self._modifyParent(node,eNode)
                if prefix == keys:
                    eNode.value = LeftNode(keys[len(prefix):],values)
                    bNode[node.keys[len(prefix)]] = node
                    node.key = node.key[len(prefix)+1:]
                elif prefix == node.keys:
                    eNode.value = node
                    node.keys = node.keys[len(prefix):]
                    bNode[keys[len(prefix)]] = LeftNode(keys[len(prefix)+1:],values)
                else:
                    bNode[keys[len(prefix)]] = LeftNode(keys[len(prefix)+1:], values)
                    bNode[node.keys[len(prefix)]] = node
                    node.key = node.key[len(prefix) + 1:]
        elif node is ExtensionNode:
            if node.nibbs == keys:
                if node.value is Node:
                    node.value = LeftNode(keys,values)
                    node.value.parent = node
                else:
                    node.value.update(values)
            elif node.nibbs.startsWith(keys):
                pass
            elif keys.startsWith(node.nibbs):
                pass
            else:
                pass

        elif node is BranchNode:
            pass





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
