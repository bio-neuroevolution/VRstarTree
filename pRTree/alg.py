
import numpy as np

def select_nodes_astar(tree,entry,nodes):
    if nodes is None or len(nodes) <=0:return None
    if nodes[0].isLeaf():
        overlop_areas = []
        for i,node in enumerate(nodes):
            mbrs = [node.mbr for j,node in enumerate(nodes) if j !=i]
            mbr = node.mbr.union(entry.mbr)
            overlop_areas.append(sum([m.overlop_area(mbr) for m in mbrs]))
        overlop_areas_index = np.argsort(overlop_areas)
        min_area = overlop_areas[overlop_areas_index[0]]
        pos = 0
        for index in overlop_areas_index:
            if abs(overlop_areas[index] - min_area)<=0:
                pos += 1
            else: break

        if pos == 0: return nodes[overlop_areas_index[0]]






def merge_nodes(tree,nodes):
    pass

def split_node(tree,node):
    pass