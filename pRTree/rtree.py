from geo import Rectangle
from utils import Configuration


class RNode:
    def __init__(self,range : Rectangle):
        self.mbr = range


class RTree:
    MAX_CHILDREN_DEFAULT_GUTTMAN = 4
    MAX_CHILDREN_DEFAULT_STAR = 4

    def __init__(self,range : Rectangle,contex : Configuration):
        self.range = range
        self.contex = contex
        self.root = RNode(range)

