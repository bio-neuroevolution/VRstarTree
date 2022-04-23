#!/usr/bin/env python
# -*- coding: utf-8 -*-

from geo import Rectangle
import geo
import alg
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
    def isLeaf(self):
        return len(self.children)<=0
    def size(self):
        return 0 if self.children is None else len(self.children)




class RTree:
    MAX_CHILDREN_DEFAULT = 16
    SELECTION_DEFAULT = 'select_rstar'
    MERGE_DEFAULT = "merge_nodes"
    SPLIT_DEFAULT = 'split_node'

    algs = {'select_rstar': alg.select_nodes_astar,'merge_nodes':alg.merge_nodes,'SPLIT_DEFAULT':alg.split_node}

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
        '''
        插入数据到节点列表中去
        :param entry Entry 数据对象
        :param nodes list of RNode 节点列表
        '''
        #执行选择节点过程，没有选中的话就创建一个，如果子节点数太多，则执行合并操作
        node = self._doSelection(entry,nodes)
        if node is None:
           node = RNode(entry.mbr, parent=None, children=[], entries=[entry])
           nodes = nodes[0].parent.addChilds(node)
        if len(nodes) > self.contex.max_children_num:
            nodes = self._doMerge(nodes)

        # 如果是叶子节点，加入数据对象，若数据对象太多，则执行分裂操作
        if node.isLeaf():
            node.addEntries(entry)
            if len(node.entries)>self.context.max_entries_num:
                cnodes = self._doSplit(node)
                node.parent.split(node,cnodes)
                return cnodes[0]
            return node

        return self._insert(entry,node.children)


    def _doSelection(self,entry,nodes):
        algName = self.contxt.select_nodes_func if self.contxt.select_nodes_func is not None else RTree.SELECTION_DEFAULT
        method = RTree.algs[algName]
        return method(self,entry,nodes)

    def _doMerge(self,nodes):
        algName = self.contxt.merge_nodes_func if self.contxt.merge_nodes_func is not None else RTree.MERGE_DEFAULT
        method = RTree.algs[algName]
        return method(self, nodes)

    def _doSplit(self,node):
        algName = self.contxt.split_node_func if self.contxt.split_node_func is not None else RTree.SPLIT_DEFAULT
        method = RTree.algs[algName]
        return method(self, node)











