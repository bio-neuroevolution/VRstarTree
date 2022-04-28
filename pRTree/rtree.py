# -*- coding: utf-8 -*-
import alg
import geo
from geo import Rectangle
from node import Entry, RNode
from utils import Configuration


class RTree:
    MAX_CHILDREN_DEFAULT = 16              #缺省最大子节点树
    SELECTION_DEFAULT = 'select_rstar'     #缺省节点选择方法名
    MERGE_DEFAULT = "merge_nodes"          #缺省合并节点方法名
    SPLIT_DEFAULT = 'split_node'           #缺省分裂节点方法名
    # 算法Map
    algs = {'select_rstar': alg.select_nodes_rstar,'merge_nodes':alg.merge_nodes_ref,'split_node':alg.split_node_rstar}

    @classmethod
    def registe(cls,name,func):
        RTree.algs[name] = func

    def __init__(self,context : Configuration,range : Rectangle=None):
        """
        R树
        :param range Rectangle 初始范围
        :param contex Configuration 配置信息
        """
        self.range = range
        self.context = context
        self.root = None
        if self.range is None:
            self.range = geo.EMPTY_RECT

    def insert(self,entry:Entry):
        '''
        插入数据
        :param entry 数据对象
        '''
        self.range = entry.mbr.union(self.range)
        if self.root is None:
            self.root = RNode(entry.mbr, parent=None, children=[], entries=[entry])
            return
        self._insert(entry,[self.root])
    def _insert(self,entry,nodes:list):
        '''
        插入数据到节点列表中去
        :param entry Entry 数据对象
        :param nodes list of RNode 节点列表
        '''
        #执行选择节点过程，没有选中的话就创建一个，如果子节点数太多，则执行合并操作
        node = self._doSelection(entry,nodes)
        if node is None:
           node = RNode(entry.mbr, parent=None, children=[], entries=[])
           if nodes[0].parent is None:
               node.addEntries(entry)
               self.root = RNode(children=nodes+[node])
               return
           nodes = nodes[0].parent.addChilds(node)
        if len(nodes) > self.context.max_children_num:
            nodes = self._doMerge(nodes)
            node = self._doSelection(entry, nodes)

        # 如果是叶子节点，加入数据对象，若数据对象太多，则执行分裂操作
        if node.isLeaf():
            node.addEntries(entry)
            if len(node.entries)>self.context.max_entries_num:
                cnodes = self._doSplit(node)
                return cnodes[0]
            return node

        return self._insert(entry,node.children)


    def _doSelection(self,entry,nodes):
        if nodes is None or len(nodes)==0:return None
        # elif len(nodes) == 1:return nodes[0]
        algName = self.context.select_nodes_func
        if algName is None or algName == '' :algName = RTree.SELECTION_DEFAULT
        method = RTree.algs[algName]
        return method(self,entry,nodes)

    def _doMerge(self,nodes):
        algName = self.context.merge_nodes_func
        if algName is None or algName == '': algName = RTree.MERGE_DEFAULT
        method = RTree.algs[algName]
        return method(self, nodes)

    def _doSplit(self,node):
        algName = self.context.split_node_func
        if algName is None or algName == '': algName = RTree.SPLIT_DEFAULT
        method = RTree.algs[algName]
        return method(self, node)











