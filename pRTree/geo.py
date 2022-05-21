
import copy
import numpy as np

class Geometry:
    DIM_EXTENSIBLE = True
    def __init__(self,dimension=2,values=None):
        self.dimension = dimension
        self.values = values
        if self.values is None:self.values = []

    def empty(self):
        return self.values is None or len(self.values)<=0

    def __getitem__(self, item):
        if not isinstance(item,int):
            return 0
        if int(item)>=len(self.values):
            return 0
        return self.values[int(item)]
    def __setitem__(self, key, value):
        if not isinstance(key, int):return
        if int(key) >= len(self.values):
            if not Geometry.DIM_EXTENSIBLE:return
            self.values = self.values+[0]*(int(key)+1-len(self.values))
        self.values[int(key)] = value

    def mbr(self):
        return None
    def union(self,*gs):
        pass

    def __str__(self):
        return str(self.values)

EMPTY = Geometry()


class Rectangle(Geometry):
    def __init__(self,dimension=2,values=None):
        self.dimension = dimension
        self.values = values
        if self.values is None: self.values = []
    def clone(self):
        return Rectangle(self.dimension,copy.copy(self.values))

    def update(self,dimension,lower,upper):
        if len(self.values)<2*(dimension+1):
            self.values = self.values + [0]*((dimension+1)*2-len(self.values))
        self.values[2*dimension] = lower
        self.values[2*dimension+1] = upper
    def boundary(self,dimension):
        if self.values is None or len(self.values)<=0:
            return None
        if dimension*2 > len(self.values):
            return None
        return [self.values[dimension*2],self.values[dimension*2+1]]
    def upper(self,dimension):
        return self.values[2*dimension+1]
    def lower(self,dimension):
        return self.values[2*dimension]
    def mbr(self):
        return self.clone()
    def rela_corss(self,dimension,pos):
        '''
        在某个维度上的交叉关系
        :param dimension int 维
        :param pos  float 维上的值
        :return 矩阵相对于pos的位置，0表示在pos左边，1表示在右边
        '''
        b = self.boundary(dimension)
        if b[0] == b[1]:
            return 0 if b[0]<pos else 1
        elif b[0] <= pos and b[1] <= pos:
            return 0
        elif b[0] > pos and b[1] > pos:
            return 1
        else:
            return 0 if abs(b[0]-pos)>=abs(b[1]-pos) else 1

    def union(self,g):
        if g is None or g.empty():
            return self.clone()
        elif self.empty():
            return g.clone()

        r = Rectangle(self.dimension)
        if isinstance(g,Rectangle):
            for i in range(self.dimension):
                r[i*2] = min(self.values[i*2],g.values[i*2])
                r[i*2+1] = max(self.values[i*2+1],g.values[i*2+1])
            return r
        elif isinstance(g,list):
            r = Rectangle(self.dimension)
            for rect in g:
                for i in range(self.dimension):
                    r[i * 2] = min(self.values[i * 2], rect.values[i * 2])
                    r[i * 2 + 1] = max(self.values[i * 2 + 1], rect.values[i * 2 + 1])
            return r
        else: return self.clone()
    @classmethod
    def unions(cls,gs):
        r = EMPTY_RECT
        if gs is None or len(gs) <= 0: return r
        r = Rectangle(dimension=gs[0].dimentsion)
        for i in range(r.dimension):
            r.update(i,np.min([g.lower(i) for g in gs]),np.max([g.upper(i) for g in gs]))
        return r
    def volume(self)->float:
        if self.empty():return 0
        v = 1.
        for i in range(self.dimension):
            v *= abs(self.values[i*2+1]-self.values[i*2])
        return v
    def volumes(cls,gs):
        if gs is None or len(gs)<=0:return 0.
        r = EMPTY_RECT
        for g in gs:
            r = r.union(g)
        return r.volume()
    def overlop(self,r):
        re = Rectangle(dimension=self.dimension)
        for i in range(self.dimension):
            lower = max(self.lower(i),r.lower(i))
            upper = min(self.upper(i),r.upper(i))
            if lower > upper:return EMPTY_RECT
            re.update(i,lower,upper)
        return re
    def isOverlop(self,r):
        for i in range(self.dimension):
            lower = max(self.lower(i), r.lower(i))
            upper = min(self.upper(i), r.upper(i))
            if lower > upper: return False
        return True

    @classmethod
    def split(cls,mbrs,dimension,value):
        """
        将mbrs所有矩形在特定维度上按照value分成两组
        :param mbrs list[Rectangle]
        :param dimension int
        :param value float
        :return (list[int],list[int])
        """
        g1,g2 = [],[]
        for index,mbr in enumerate(mbrs):
            b = mbr.boundary(dimension)
            if b[1]<=value: g1.append(index)
            elif b[0] >= value: g2.append(index)
            elif abs(b[0]-value) >= abs(b[1]-value):
                g1.append(index)
            else:
                g2.append(index)
        return (g1,g2)


    @classmethod
    def groupby(cls,tree, mbrs, mode='overlop_area'):
        """
        将mbrs分成两组
        :param tree RTree
        :param mbrs list[Rectangle]
        :param mode str 分组方式 'area' 最小面积 'overlop'最小重叠面积 'overlop_area' 先最小重叠面积，多个再用最小面积
        :return (mbr, d, i, area, overlop, g1, g2) 最优分组 g1和g2是分组的索引
        """
        plans, min_overlop, min_area = [], 0, 0
        for mbr in mbrs:
            for d in range(mbr.dimension):
                bound = mbr.boundary(d)
                for i in range(2):
                    g1 = [m for m in mbrs if m.rela_corss(d, bound[i]) == 0 and m != mbr]
                    if i == 1: g1.append(mbr)
                    g2 = list(set(mbrs) - set(g1))
                    if len(g1) <= 0 or len(g2) <= 0:
                        continue
                    mbr1 = Rectangle.unions([g for g in g1])
                    mbr2 = Rectangle.unions([g for g in g2])
                    area = mbr1.volume() + mbr2.volume()
                    overlop = mbr1.overlop(mbr2).volume()
                    plans.append(dict(mbr=mbr,dimension=d,index=i,area=area,overlop=overlop,g=[g1,g2]))
        # 如果没有得到任何方案，则平均分成两组
        if len(plans) == 0:
            mbr = mbrs[len(mbrs)//2]
            g1 = mbrs[:len(mbrs)//2]
            g2 = mbrs[len(mbrs)//2:]
            mbr1, mbr2 = Rectangle.unions(g1), Rectangle.unions(g2)
            overlop = mbr1.overlop(mbr2).volume()
            area = mbr1.volume() + mbr2.volume()
            return  (mbr, -1, -1, area, overlop, [mbrs.index(m) for m in g1], [mbrs.index(m) for m in g2])

        if mode is None or mode == '':
            for p in plans:
                g1,g2 = p['g'][0],p['g'][1]
                p['g'][0],p['g'][1] = [mbrs.index(m) for m in g1], [mbrs.index(m) for m in g2]
            return plans

        optima, min_area = None, 0
        if mode == 'overlop':
            index = np.argmin([p['overlop'] for p in plans])
            optima = plans[index]
        elif mode == 'area':
            index = np.argmin([p['area'] for p in plans])
            optima = plans[index]
        elif mode == 'overlop_area':
            index = np.argmin([p['overlop'] for p in plans])
            min_overlop = plans[index]['overlop']
            plans = [p for p in plans if abs(p['overlop']-min_overlop)<=0]
            index = np.argmin([p['area'] for p in plans])
            optima = plans[index]
        else:
            optima = plans[0]

        mbr, d, i, area, overlop, g1, g2 = optima['mbr'],optima['dimension'],optima['index'],optima['area'],optima['overlop'],optima['g'][0],optima['g'][1]
        return (mbr, d, i, area, overlop, [mbrs.index(m) for m in g1], [mbrs.index(m) for m in g2])

EMPTY_RECT = Rectangle()


class Point(Rectangle):
    def __init__(self,dimension=2,values=None):
        super(Rectangle,self).__init__(dimension,values)
    def mbr(self):
        return self.clone()
    def __getattr__(self, item):
        if str(item).lower() == 'x': return self.values[0]
        elif str(item).lower() == 'y': return self.values[1]
        elif str(item).lower() == 'z':return self.values[2]
        elif isinstance(item,int): return self.values[int(item)]
        else: return  None
    def __setattr__(self, key, value):
        if str(key).lower() == 'x':
            if len(self.values)<=0: self.values = [0]
            self.values[0] = value
        elif str(key).lower() == 'y':
            if len(self.values)<=1: self.values = self.values+[0]*(2-len(self.values))
            self.values[1] = value
        elif str(key).lower() == 'z':
            if len(self.values)<=2: self.values = self.values+[0]*(3-len(self.values))
            self.values[2] = value
        elif isinstance(key, int):
            if len(self.values) <= int(key)+1: self.values = self.values + [0] * (3 - len(self.values))
    def __getitem__(self, item):
        return self.__getattr__(item)
    def __setitem__(self, key, value):
        self.__setattr__(key,value)



