
import copy

class Geometry:
    DIM_EXTENSIBLE = True
    def __init__(self,dimension=2,values=None):
        self.dimension = dimension
        self.values = values
        if self.values is None:self.values = []
    def clone(self):
        return copy.deepcopy(self)
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
        super(Rectangle,self).__init__(dimension,values)
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
        if g is None:
            return self.clone()
        elif g.empty():
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
        for g in gs:
            r = r.union(g)
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
        return self.overlop(r) is not None

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



