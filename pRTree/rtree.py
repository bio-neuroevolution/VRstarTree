# -*- coding: utf-8 -*-
import math
import queue


import alg
import geo
from geo import Rectangle
from node import Entry, RNode
from utils import Configuration
from utils import Collections
from graphviz import Digraph, nohtml


class RTree:
    MAX_CHILDREN_DEFAULT = 16              #缺省最大子节点树
    SELECTION_DEFAULT = 'select_rstar'     #缺省节点选择方法名
    MERGE_DEFAULT = "merge_nodes"          #缺省合并节点方法名
    SPLIT_DEFAULT = 'split_node'           #缺省分裂节点方法名
    # 算法Map
    algs = {'select_rstar': alg.select_nodes_rstar,'merge_nodes':alg.merge_nodes_rstar,'split_node':alg.split_node_rstar}

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
        self.query_node_count = 0
        self.depth = 0

    def serialize(self):
        return {'range':str(self.range),'root':RNode.serialize(self.root)}

    def count(self):
        if self.root is None:return 0
        value = [1]
        self._count(self.root,value)
        return value[0]
    def _count(self,node,value):
        value[0] += len(node.children)
        if len(node.children) <= 0:
            value[0] += len(node.entries)
            return

        for cnode in node.children:
           self._count(cnode,value)
        return value[0]


    def insert(self,entry:Entry):
        '''
        插入数据
        :param entry 数据对象
        '''
        self.range = entry.mbr if self.range.empty() else entry.mbr.union(self.range)
        if self.root is None:
            self.root = RNode(entry.mbr, parent=None, children=[], entries=[entry])
            self.depth = 1
            return
        self._insert(entry,self.root)

    def _insert(self, entry, node):
        '''
        插入数据到节点列表中去
        :param entry Entry 数据对象
        :param nodes list of RNode 节点列表
        '''
        node = self._doSelection(entry,node)
        node.addEntries(entry)
        if len(node.entries) > self.context.max_entries_num:
            cnodes = self._doSplit(node)

    def _insert2(self,entry,nodes:list):
        '''
        插入数据到节点列表中去
        :param entry Entry 数据对象
        :param nodes list of RNode 节点列表
        '''
        #执行选择节点过程，没有选中的话就创建一个，如果子节点数太多，则执行合并操作
        node = self._doSelection(entry,nodes)
        if node is None:
           node = RNode(entry.mbr, parent=None, children=[], entries=[entry])
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
                if len(cnodes)> self.context.max_children_num:
                    cnodes = self._doMerge(cnodes)
                return cnodes[0]
            return node

        return self._insert(entry,node.children)




    def find2(self, mbr: Rectangle):
        if self.root is None : return []
        if isinstance(self.root, list) and len(self.root) == 0: return []
        self.query_node_count = 0
        if not self.root.mbr.isOverlop(mbr): return []
        q = queue.SimpleQueue()
        q.put(self.root)
        results,level = [],0
        num = 0
        while not q.empty():
            num += 1
            node = q.get_nowait()
            if node.isLeaf():
                if node.mbr.isOverlop(mbr):
                    node.ref += 1
                    self.query_node_count += len(node.entries)
                    rs = [entry for entry in node.entries if entry.mbr.isOverlop(mbr)]
                    for r in rs: r.ref += 1
                    results += rs
            else:
                for cnode in node.children:
                    if not cnode.mbr.isOverlop(mbr):continue
                    self.query_node_count += 1
                    cnode.ref += 1
                    q.put_nowait(cnode)

        #print('query iter num='+str(num))
        return results


    def find(self,mbr:Rectangle,node:RNode=None):
        if mbr is None or mbr.empty():
            return []
        if self.root is None: return []
        if isinstance(self.root,list) and len(self.root)==0:return []
        if node is None:
            self.query_node_count = 0
            return self.find(mbr,self.root)

        cross = node.mbr.overlop(mbr)
        if cross.empty():return []

        if node.isLeaf():
            if node.mbr.isOverlop(mbr):
                node.ref += 1
                self.query_node_count += len(node.entries)
                rs = [entry for entry in node.entries if entry.mbr.isOverlop(mbr)]
                for r in rs: r.ref += 1
                return rs

        rs = []
        for cnode in node.children:
            corss = cnode.mbr.overlop(mbr)
            if corss.empty():continue
            cnode.ref += 1
            self.query_node_count += 1
            rs = rs + self.find(mbr,cnode)

        return rs

    def proof_length(self, count, unit):
        return unit*self.depth

    def _doSelection(self,entry,nodes):
        #if nodes is None or len(nodes)==0:return None
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

    def all_entries(self,node=None):
        if node is None:
            return self.all_entries(self.root)

        if len(node.entries)>0:
            return node.entries
        if len(node.children)<=0: return []

        r = []
        for cnode in node.children:
            r = r + self.all_entries(cnode)
        return  r


    def rearrage_all3(self):
        oldalgs = RTree.algs.copy()
        RTree.algs['merge_nodes'] = alg.merge_nodes_ref
        RTree.algs['split_node'] = alg.split_node_ref

        q = queue.SimpleQueue()
        q.put(self.root)
        #self.context.max_entries_num = 8
        while not q.empty():
            node = q.get_nowait()
            if node.isLeaf():continue
            if len(node.children)<=0:continue
            for cnode in node.children:
                q.put_nowait(cnode)
            if node.children[0].isLeaf():
                entries = []
                for cnode in node.children:
                    entries += cnode.entries
                entries = sorted(entries,key=lambda e:e.ref,reverse=True)
                size = math.ceil(len(entries) / self.context.max_entries_num)
                node.children = []
                children = []
                for i in range(size):
                    children.append(RNode(parent=node,children=[],entries=entries[i*self.context.max_entries_num:(i+1)*self.context.max_entries_num]))

                if len(node.children)>self.context.max_children_num:
                    self._doMerge(node.children)
        RTree.algs = oldalgs

    def rearrage_nodes(self,nodes,results=[]):
        mbrs = [n.mbr for n in nodes]
        plans, maxcov, minarea, minoverlop = [], 0,0,0
        for i, node in enumerate(nodes):
            for d in range(node.mbr.dimension):
                b = node.mbr.boundary(d)
                for j in range(2):
                    g1, g2 = Rectangle.split(mbrs,d, b[j])
                    if len(g1)<=0 or len(g2)<=0:continue
                    g1, g2 = [nodes[g] for g in g1], [nodes[g] for g in g2]
                    mbr1, mbr2 = Rectangle.unions([g.mbr for g in g1]), Rectangle.unions([g.mbr for g in g2])
                    overlop = mbr1.overlop(mbr2).volume()
                    area = mbr1.volume()+mbr2.volume()
                    cov = Collections.group_cov([g.ref for g in g1], [g.ref for g in g2])
                    plans.append(dict(g=[g1, g2], cov=cov,node=node, d=d, pos=j,overlop=overlop, area=area))

        if len(plans) == 0:
            g1 = nodes[:len(nodes)//2]
            g2 = nodes[len(nodes)//2:]
            mbr1, mbr2 = Rectangle.unions([g.mbr for g in g1]), Rectangle.unions([g.mbr for g in g2])
            overlop = mbr1.overlop(mbr2).volume()
            area = mbr1.volume() + mbr2.volume()
            cov = Collections.group_cov([g.ref for g in g1], [g.ref for g in g2])
            plans.append(dict(g=[g1,g2], cov=cov,node=node, d=d, pos=j,overlop=overlop, area=area))

        # 选择组间方差大的
        plans = sorted(plans, key=lambda p: p['cov'], reverse=True)
        maxcov = plans[0]['cov']
        plans = [p for p in plans if abs(maxcov - p['cov']) <= 5.0]

        #选择总面积小的
        plans = sorted(plans, key=lambda p: p['area'])
        minarea = plans[0]['area']
        plans = [p for p in plans if abs(p['area']-minarea)<=0]

        #最后选择重叠面积小的
        plans = sorted(plans,key=lambda p:p['overlop'])
        optima = plans[0]
        for i in range(2):
            g = optima['g'][i]
            if len(g)<=self.context.max_children_num:
                results.append(g)
            else:
                self.rearrage_nodes(g,results)

    def rearrage_leafs(self,entries,leafs):
        """
        根据数据对象重新安排所有的叶子节点
        """
        if len(entries) == 0: return []

        plans, maxcov, minarea, minoverlop = [], 0,0,0
        mbrs = [e.mbr for e in entries]
        for i, entry in enumerate(entries):
            for d in range(entry.mbr.dimension):
                b = entry.mbr.boundary(d)
                for j in range(2):
                    if j == 1 and b[0] == b[1]:break
                    g1, g2 = Rectangle.split(mbrs,d, b[j])
                    if len(g1)<=0 or len(g2)<=0:continue
                    g1, g2 = [entries[g] for g in g1],[entries[g] for g in g2]
                    mbr1, mbr2 = Rectangle.unions([g.mbr for g in g1]), Rectangle.unions([g.mbr for g in g2])
                    cov = Collections.group_cov([g.ref for g in g1], [g.ref for g in g2])
                    plans.append(dict(g=[g1, g2], cov=cov, entry=entry, d=d, pos=j,
                                          overlop=mbr1.overlop(mbr2).volume(), area=mbr1.volume() + mbr2.volume()))

        # 选择组间方差大的
        plans = sorted(plans,key=lambda p:p['cov'],reverse=True)
        maxcov = plans[0]['cov']
        plans = [p for p in plans if abs(maxcov-p['cov'])<=5.0]

        #选择重叠面积小的
        plans = sorted(plans, key=lambda p: p['overlop'])
        optima = plans[0]
        for i in range(2):
            if len(optima['g'][i]) <= self.context.max_entries_num:
                leafs.append(RNode(mbr=None,parent=None, children=[], entries=optima['g'][i]))
            else:
                self.rearrage_leafs(optima['g'][i],leafs)
        return leafs

    def rearrage_all(self):

        oldalgs = RTree.algs.copy()
        RTree.algs['split_node'] = alg.split_node_ref

        entries = self.all_entries()
        nodes = []
        self.rearrage_leafs(entries,nodes)
        results = []
        while True:
            self.rearrage_nodes(nodes,results)
            parents = [RNode(mbr=None,parent=None,children=r,entries=[]) for r in results]
            if len(parents)<=self.context.max_children_num:
                self.root = RNode(mbr=None,parent=None,children=parents,entries=[])
                return
            results,nodes = [],parents


        RTree.algs = oldalgs




    def rearrage_all2(self):
        entries = self.all_entries()
        oldalgs = RTree.algs.copy()
        RTree.algs['merge_nodes'] = alg.merge_nodes_ref
        RTree.algs['split_node'] = alg.split_node_ref

        entries = sorted(entries,key=lambda e:e.ref,reverse=True)
        self.context.max_entries_num = 8
        self.root = None
        for entry in entries:
            self.insert(entry)

        RTree.algs = oldalgs

    def view(self,path,filename):
        g = Digraph('g', filename='rtree.gv', node_attr={'shape': 'record', 'height': '.1'})
        self._create_viewnode(g,self.root)
        g.view(filename=filename, directory=path)

    def _create_viewnode(self,g,node,parentId=None,id=1):
        if node is None: return
        if len(node.children)>0:
            content = ''
            for i in range(len(node.children)):
                if node.children[i] is None:
                    continue
                content = content + '<f' + str(i) + '>' + chr(i) + '|'
            content = content[:-1]
            gid = str(id)
            gn = self.g.node(id, nohtml(content))
            if parentId is not None:
                self.g.edges([(parentId, gid)])

            for n in node.children:
                id += 1
                self._create_viewnode(g,n,gid,id)
        else:
            content = ''
            for i in range(len(node.entries)):
                content = content + '<f' + str(i) + '>' + chr(i) + '|'
            content = content[:-1]
            gid = str(id)
            gn = self.g.node(id, nohtml(content))
            if parentId is not None:
                self.g.edges([(parentId, gid)])
















