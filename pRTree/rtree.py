#!/usr/bin/env python
# -*- coding: utf-8 -*-

from geo import Rectangle
import geo
import alg
from utils import Configuration, Collections
import numpy as np

class Entry:
    def __init__(self,id,mbr,datas):
        """
        数据对象
        id Union(str,int)  对象id
        mbr Rectangle      对象的空间范围
        datas any         数据信息
        """
        self.id = id
        self.mbr = mbr
        self.datas = datas
class RNode:
    def __init__(self,mbr : Rectangle,parent=None,children : list=[],entries:list=[]):
        '''
        R树节点
        mbr Rectangle 时空范围
        parent RNode 父节点
        children list of RNode 所有子节点
        entries list of Entry 所有数据对象，只有叶节点才有
        '''
        self.mbr = mbr
        self.parent = parent
        self.children = children
        self.entries = entries
        if self.mbr is None: self.mbr = geo.EMPTY_RECT
        if len(children)>0:
            Collections.assign('parent',self,self.children)
            self.mbr = self.mbr.union([node.mbr for node in self.children])
        if self.parent is not None:
            self.parent.children.apppend(self)
    def addEntries(self,*entries):
        """
        添加数据
        entries list of Entry 数据对象集
        """
        if entries is None or len(entries)<=0:return
        newentries = entries - self.entries
        self.entries = self.entries + newentries
        self._update_mbr()
    def getEntries(self,ids):
        '''
        查询指定Id的数据
        :param ids Union(int,str,list)
        '''
        if ids is None:return None
        if isinstance(ids,list):
            return [self.getEntries(e) for e in ids]
        else:
            for e in self.entries:
                if e.id == ids: return e
            return None
    def clearEntries(self):
        '''
        清空数据集
        '''
        self.entries = []

    def _update_mbr(self):
        '''
        更新Mbr数据，同时更新父节点的mbr数据
        '''
        if len(self.entries)>0:
            self.mbr = Rectangle.unions([entry.mbr for entry in self.entries])
        else:
            self.mbr = Rectangle.unions([n.mbr for n in self.children])
        if self.parent is None: return
        self.parent._update_mbr()

    def addChilds(self,*nodes):
        """
        添加子节点
        :param nodes list
        """
        if nodes is None or len(nodes)<=0:return
        newnodes = []
        for n in nodes:
            if not self.contains(n):
                newnodes.append(n)
                n.parent = self
        self.children = self.children + newnodes
        self._update_mbr()
        return self.children

    def contains(self,nodesOrEntries):
        '''
        是否包含子节点或者数据对象
        :param nodesOrEntries Union(RNode,Entry,int,str)
        '''
        if nodesOrEntries is None:return False
        if isinstance(nodesOrEntries,list):
            return [self.contains(e) for e in nodesOrEntries]
        elif isinstance(nodesOrEntries,RNode):
            return nodesOrEntries in self.children
        elif isinstance(nodesOrEntries,Entry):
            return nodesOrEntries in self.entries
        else:
            return nodesOrEntries in [e.id for e in self.entries]


    def isLeaf(self):
        """
        是否是叶子节点
        """
        return len(self.children)<=0
    def size(self,type):
        '''
        叶子节点数量
        :param type Union(int,str) 1或者'entry'为数据，2或者'node'为子节点
        '''
        if type is None or type == 2 or type == 'node':
            return len(self.children)
        else:
            return len(self.entries)



class RTree:
    MAX_CHILDREN_DEFAULT = 16              #缺省最大子节点树
    SELECTION_DEFAULT = 'select_rstar'     #缺省节点选择方法名
    MERGE_DEFAULT = "merge_nodes"          #缺省合并节点方法名
    SPLIT_DEFAULT = 'split_node'           #缺省分裂节点方法名
    # 算法Map
    algs = {'select_rstar': alg.select_nodes_astar,'merge_nodes':alg.merge_nodes_ref,'SPLIT_DEFAULT':alg.split_node_rstar}

    def __init__(self,contex : Configuration,range : Rectangle=None):
        """
        R树
        :param range Rectangle 初始范围
        :param contex Configuration 配置信息
        """
        self.range = range
        self.contex = contex
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
           node = RNode(entry.mbr, parent=None, children=[], entries=[entry])
           nodes = nodes[0].parent.addChilds(node)
        if len(nodes) > self.contex.max_children_num:
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











