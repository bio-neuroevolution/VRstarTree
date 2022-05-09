
import numpy as np
import geo
import sys

from node import RNode
from utils import Collections

def select_nodes_rstar(tree,entry,node):
    """
    选择节点的R*-tree算法
    :param tree RTree  R树
    :param entry Entry 数据
    :param node  在该节点的子节点一种找到一个合适的叶节点
    """
    # 叶节点已找到，直接返回
    if node.isLeaf(): return node
    # 不是叶子节点，且只有一个子节点
    if len(node.children)==1:
        return node.children[0] if node.children[0].isLeaf() else select_nodes_rstar(tree,entry,node.children[0])

    # 遍历所有子节点选择一个合适的
    plans,min_overlop_area,min_area = [],0,0

    for i, cnode in enumerate(node.children):
        mbrs = [node.mbr for j, node in enumerate(node.children) if j != i]
        mbr = node.mbr.union(entry.mbr)
        overlop_area = sum([m.overlop(mbr).volume() for m in mbrs])
        area = mbr.volume() - node.mbr.volume()

        if len(plans) == 0:
            plans.append({'index':i,'overlop_area':overlop_area,'area':area})
        elif node.children[0].isLeaf() and abs(min_overlop_area-overlop_area)<=0:
            plans.append({'index': i, 'overlop_area': overlop_area, 'area': area})
            min_overlop_area = overlop_area
            min_area = area
        elif not node.children[0].isLeaf() and abs(min_area - area)<=0:
            plans.append({'index': i, 'overlop_area': overlop_area, 'area': area})
            min_overlop_area = overlop_area
            min_area = area

    # 只得到一个方案
    if len(plans)==1:
        if node.children[0].isLeaf():
            return node.children[plans[0]['index']]
        else:
            return select_nodes_rstar(tree,entry,node.children[plans[0]['index']])
    # 多个方案，且子节点不是叶子节点
    if not node.children[0].isLeaf():
        return select_nodes_rstar(tree,entry,node.children[plans[0]['index']])

    # 多个方案，且子节点是叶子节点
    optima,min_area = None,0
    for plan in plans:
        if optima is None or min_area > plan['area']:
            optima,min_area = plan,plan['area']

    return node.children[optima['index']]


def select_nodes_level(tree,entry,nodes):
    """
    选择节点的R*-tree算法
    :param tree RTree  R树
    :param entry Entry 数据
    :param nodes list of RNode 待选择的节点列表
    """
    if nodes is None or len(nodes) <=0:return None
    if len(nodes) == 1:
        return nodes[0] if nodes[0].mbr.isOverlop(entry.mbr) else None

    # 如果是叶节点
    if nodes[0].isLeaf():
        # 尝试将数据加入到每个节点，计算加入后与其它节点的重叠面积和增长面积
        overlop_areas,areas = [],[]
        for i,node in enumerate(nodes):
            mbrs = [node.mbr for j,node in enumerate(nodes) if j !=i]
            mbr = node.mbr.union(entry.mbr)
            overlop_areas.append(sum([m.overlop(mbr).volume() for m in mbrs]))
            areas.append(mbr.volume()-node.mbr.volume())
        # 所有数据加入方案的重叠面积从小到大排序
        overlop_areas_index = np.argsort(overlop_areas)
        # 所有数据加入方案的增加面积从小到大排序
        min_area = overlop_areas[overlop_areas_index[0]]
        # 取得所有数据加入方案重叠面积较小的那些（从0到pos）
        pos = 0
        for index in overlop_areas_index:
            if abs(overlop_areas[index] - min_area)<=0:
                pos += 1
            else: break
        if pos == 0: return nodes[overlop_areas_index[0]]
        overlop_areas_index = overlop_areas_index[:pos]
        # 在剩余的方案中选择面积增加最小的
        area,rindex = areas[overlop_areas_index[0]],0
        for index in overlop_areas_index:
            if area > areas[index]:
                area = areas[index]
                rindex = index
        return nodes[rindex]
    else:
        # 尝试将数据加入到每个节点，计算加入后与其它节点的重叠面积和增长面积
        areas = []
        for i, node in enumerate(nodes):
            mbr = node.mbr.union(entry.mbr)
            areas.append(mbr.volume() - node.mbr.volume())
        # 所有数据加入方案的重叠面积从小到大排序
        am = np.argmin(areas)
        return nodes[am]

def merge_nodes_ref(tree,nodes):
    '''
    根据引用计数合并节点
    '''
    if nodes is None or len(nodes)<=0:return nodes
    ## 还不需要合并
    if len(nodes)<=tree.context.max_children_num:
        return nodes
    ## 按照引用次数从大到小排序
    nodes = sorted(nodes,key=lambda n: n.ref,reverse=True)

    areas,overlops = [],[]
    for i in range(len(nodes)-1):
        m1 = geo.EMPTY_RECT
        for node in nodes[:i+1]:
            m1 = m1.union(node.mbr)
        m2 = geo.EMPTY_RECT
        for node in nodes[i+1:]:
            m2 = m2.union(node.mbr)
        areas.append(m1.volume()+m2.volume())
        #overlops.append(m1.intersection(m2).volumes())
    index = np.argmin(areas)

    parent = nodes[0].parent
    parent.children = []
    p1 = RNode(parent=nodes[0].parent,children=nodes[:i+1],entries=[])
    p2 = RNode(parent=nodes[0].parent,children=nodes[i+1:],entries=[])
    parent._update_mbr()

    if parent.parent is not None:
        if len(parent.parent.children) > tree.context.max_children_num:
            return merge_nodes_ref(tree, parent.parent.children)
    else:
        return parent.children


def merge_nodes_rstar_combinations(tree, nodes):
    '''
    根据引用计数合并节点
    '''
    if nodes is None or len(nodes) <= 0: return nodes
    ## 还不需要合并
    if len(nodes) <= tree.context.max_children_num:
        return nodes

    plan, minarea = None, 0
    for n in range(len(nodes) - 1):
        groups = Collections.combinations(nodes, n + 1)
        for group in groups:
            g1 = group
            g2 = list(set(nodes) - set(g1))

            m1 = geo.Rectangle.unions([g.mbr for g in g1])
            m2 = geo.Rectangle.unions([g.mbr for g in g2])
            area = m1.volume() + m2.volume()
            if plan is None or minarea > area:
                plans = (g1, g2, m1, m2, minarea)
                minarea = area

    parent = nodes[0].parent
    parent.children = []
    p1 = RNode(parent=nodes[0].parent, children=g1, entries=[])
    p2 = RNode(parent=nodes[0].parent, children=g2, entries=[])
    parent._update_mbr()

    # if parent.parent is not None:
    #    if len(parent.parent.children) > tree.context.max_children_num:
    #        return merge_nodes_rstar(tree, parent.parent.children)
    # else:
    return parent.children

def merge_nodes_rstar(tree,nodes):
    '''
    根据引用计数合并节点
    '''
    if nodes is None or len(nodes)<=0:return nodes
    ## 还不需要合并
    if len(nodes)<=tree.context.max_children_num:
        return nodes

    optima = _groupby(tree,[node.mbr for node in nodes],'area')
    if optima is None:
        return nodes

    g1 = [nodes[g] for g in optima[5]]
    g2 = [nodes[g] for g in optima[6]]

    parent = nodes[0].parent
    parent.children = []
    p1 = RNode(parent=nodes[0].parent,children=g1,entries=[])
    p2 = RNode(parent=nodes[0].parent,children=g2,entries=[])
    parent._update_mbr()

    return parent.children

def _groupby(tree,mbrs,mode='overlop_area'):
    """
    将mbrs分成两组
    :param tree RTree
    :param mbrs list[Rectangle]
    :param mode str 分组方式 'area' 最小面积 'overlop'最小重叠面积 'overlop_area' 先最小重叠面积，多个再用最小面积
    :return (mbr, d, i, area, overlop, g1, g2) 最优分组 g1和g2是分组的索引
    """
    plans, min_overlop,min_area = [], 0
    for mbr in mbrs:
        for d in range(mbr.dimension):
            bound = mbr.boundary(d)
            for i in range(2):
                g1 = [m for m in mbrs if m.rela_corss(d,bound[i])==0 and m != mbr]
                if i == 1: g1.append(mbr)
                g2 = list(set(mbrs) - set(g1))
                if len(g1)<=0 or len(g2)<=0:
                    continue
                mbr1 = geo.Rectangle.unions([g.mbr for g in g1])
                mbr2 = geo.Rectangle.unions([g.mbr for g in g2])
                area = mbr1.volume()+mbr2.volume()
                overlop = mbr1.overlop(mbr2).volume()
                if len(plans) <= 0:
                    plans.append((mbr, d, i, area, overlop, g1, g2))
                    min_overlop,min_area = overlop,area
                elif (mode == 'overlop' or mode=='overlop_area') and abs(min_overlop - overlop)<=0:
                    plans.append((mbr, d, i, area, overlop, g1, g2))
                    min_overlop, min_area = overlop, area
                elif mode == 'area' and abs(min_area - area)<=0:
                    plans.append((mbr, d, i, area, overlop, g1, g2))
                    min_overlop, min_area = overlop, area
    optima,min_area = None,0
    if len(plans)==1 or mode != 'overlop_area':
        optima = plans[0]
    else:
        for p in plans:
            _, _, _, area, _, _, _ = p
            if minarea == 0 or minarea > area:
                minarea, optima = area, p

    optima[5] = [mbrs.inex(m) for m in g1]
    optima[6] = [mbrs.inex(m) for m in g2]

    return optima

def split_node_rstar(tree,node):
    '''
    R*-tree的分裂算法
    :param tree RTree
    :param node RNode 一定是叶节点
    '''
    if len(node.entries)<=0:return [node]
    # 对每个数据对象的，每个轴，两个边界上分别尝试分裂，保留重叠面积较小的方案，然后选择其中总面积最小的方案
    plans,min_overlop = [],0
    for entry in node.entries:
        for d in range(entry.mbr.dimension):
            bound = entry.mbr.boundary(d)
            for i in range(2):
                g1 = [e for e in node.entries if e.mbr.rela_corss(d,bound[i])==0 and e != entry]
                if i == 1: g1.append(entry)
                g2 = list(set(node.entries) - set(g1))
                if len(g1)<=0 or len(g2)<=0:
                    continue
                mbr1 = geo.Rectangle.unions([g.mbr for g in g1])
                mbr2 = geo.Rectangle.unions([g.mbr for g in g2])
                area = mbr1.volume()+mbr2.volume()
                overlop = mbr1.overlop(mbr2).volume()
                if len(plans)<=0 or abs(min_overlop - overlop)<=0:
                    plans.append((entry,d,i,area,overlop,g1,g2))
                    min_overlop = overlop
    minarea,optima = 0,None
    for p in plans:
        _,_,_,area,_,_,_ = p
        if minarea == 0 or minarea > area:
            minarea,optima = area,p

    entry,dimension,bound,area,overlop,g1,g2 = optima
    if node.parent is None:
        tree.root = RNode()
        tree.root.children = [RNode(parent=tree.root,children=[],entries=g1),RNode(parent=tree.root,children=[],entries=g2)]
        tree.root._update_mbr()
        return tree.root.children
    else:
        node.parent.children = list(set(node.parent.children) - set([node]))
        parent,node.parent = node.parent,None
        n1 = RNode(parent=parent,children=[],entries=g1)
        n2 = RNode(parent=parent,children=[],entries=g2)
        parent._update_mbr()
        return parent.children

def split_node_ref(tree,node):
    '''
    基于引用计数的分裂算法
    :param tree RTree
    :param node RNode 一定是叶节点
    '''
    if len(node.entries)<=0:return node.parent.children

    #引用计数降序
    refs = np.array([entry.ref for entry in node.entries])
    refindex = np.argsort(-np.array(refs))

    totoal_average = np.average(refs)
    maxcov,plans = 0,[]
    for i in range(len(refindex)-1):
        r1 = [refs[index] for index in refindex[:i+1]]
        r2 = [refs[index] for index in refindex[i+1:]]
        g1 = [node.entries[index] for index in refindex[:i+1]]
        g2 = [node.entries[index] for index in refindex[i+1:]]
        g1_avg = np.average(r1)
        g2_avg = np.average(r2)


        cov = (((g1_avg-totoal_average)**2)*len(r1) + ((g2_avg-totoal_average)**2)*len(r2))/len(refs)
        if len(plans)<=0 or abs(maxcov-cov)<=0:
            maxcov = cov
            plans.append((i,g1,g2,cov))

    optima,minarea = None,0
    if len(plans)==1:
        optima = plans[0]
    else:
        for plan in plans:
            i,g1,g2,cov = plan
            mbr1 = geo.Rectangle.unions([g.mbr for g in g1])
            mbr2 = geo.Rectangle.unions([g.mbr for g in g2])
            area = mbr1.volume() + mbr2.volume()
            #overlop = mbr1.overlop(mbr2).volume()
            if optima is None or minarea > area:
                optima,minarea = plan,area

    if node.parent is None:
        tree.root = RNode()
        tree.root.children = [RNode(parent=tree.root,children=[],entries=g1),RNode(parent=tree.root,children=[],entries=g2)]
        return tree.root.children
    else:
        node.parent.children = list(set(node.parent.children) - set([node]))
        parent,node.parent = node.parent,None
        n1 = RNode(parent=parent,children=[],entries=g1)
        n2 = RNode(parent=parent,children=[],entries=g2)
        return parent.children












