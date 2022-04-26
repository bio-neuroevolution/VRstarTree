from geo import Rectangle
import geo
from utils import Configuration, Collections


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


