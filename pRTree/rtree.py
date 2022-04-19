from geo import Rectangle
from utils import Configuration

class Entry:
    def __init__(self,id,mbr,datas):
        self.id = id
        self.mbr = mbr
        self.datas = datas
class RNode:
    def __init__(self,range : Rectangle,parent=None,children : list=[]):
        self.mbr = range
        self.parent = parent
        self.children = children

class LeafNode(RNode):
    def __init__(self,range : Rectangle,parent,children : list,entries):
        super(LeafNode, self).__init__(range, parent,children)
        self.entries = entries

class RTree:
    MAX_CHILDREN_DEFAULT_GUTTMAN = 4
    MAX_CHILDREN_DEFAULT_STAR = 4

    def __init__(self,range : Rectangle,contex : Configuration):
        self.range = range
        self.contex = contex
        self.root = None

    def insert(self,entry):
        self.range = entry.mbr.union(self.range)
        if self.root is None:
            self.root = LeafNode(entry.mbr, parent=None, children=[], entries=[entry])
            return
        self._insert(entry,[self.root])
    def _insert(self,entry,nodes):
        intersections = Rectangle.intersections(entry.mbr,[node.mbr for node in nodes])
        hasIntersections = [not inter.isEmpty() for inter in intersections]







