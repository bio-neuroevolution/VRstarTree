from geo import Rectangle
import geo
from utils import Configuration, Collections
import numpy as np

class Entry:
    def __init__(self,id,mbr,datas):
        self.id = id
        self.mbr = mbr
        self.datas = datas
class RNode:
    def __init__(self,range : Rectangle,parent=None,children : list=[],entries:list=[]):
        self.mbr = range
        self.parent = parent
        self.children = children
        self.entries = entries
        if self.mbr is None: self.mbr = geo.EMPTY_RECT
        if len(children)>0:
            Collections.assign('parent',self,self.children)
            self.mbr = self.mbr.union([node.mbr for node in self.children])
    def addEntries(self,*entries):
        if entries is None or len(entries)<=0:return
        newentries = []
        for n in entries:
            if n not in self.entries:
                newentries.append(n)
        self.mbr = self.mbr.union([entry.mbr for entry in entries])
        self.entries = self.entries + newentries
    def addChilds(self,*nodes):
        if nodes is None or len(nodes)<=0:return
        newnodes = []
        for n in nodes:
            if not self.contains(n):
                newnodes.append(n)
                n.parent = self
        self.mbr = self.mbr.union([node.mbr for node in newnodes])
        self.children = self.children + newnodes
        return self.children
    def contains(self,node):
        return node in self.children
    def size(self):
        return 0 if self.children is None else len(self.children)



class RTree:
    MAX_CHILDREN_DEFAULT = 16
    SELECTION_DEFAULT_ = 'select'

    def __init__(self,range : Rectangle,contex : Configuration):
        self.range = range
        self.contex = contex
        self.root = None

    def insert(self,entry):
        self.range = entry.mbr.union(self.range)
        if self.root is None:
            self.root = RNode(entry.mbr, parent=None, children=[], entries=[entry])
            return
        self._insert(entry,[self.root])
    def _insert(self,entry,nodes:list):
        if nodes is None or len(nodes)<=0:
            self.root = RNode(entry.mbr, parent=None, children=[], entries=[entry])
            return

        node = self._doSelection(entry,nodes)
        if node is None:
           node = RNode(entry.mbr, parent=None, children=[], entries=[entry])

        nodes = nodes[0].parent.addChilds(node)
        if len(nodes) > self.contex.max_children_num:
            nodes = self._doMerge(nodes)

        if node.size()<=0: #叶子节点





        scores = self._score(entry,nodes,len(nodes)>=self.contex.max_children_num)
        scoreIndexes = np.argmax(scores)
        overlaps = Rectangle.intersections(entry.mbr,[node.mbr for node in nodes])
        hasoverlaps = [not overlap.isEmpty() for overlap in overlaps]
        if Collections.all(lambda d:not d,hasoverlaps): #没有重叠








