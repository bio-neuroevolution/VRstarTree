
import copy

class Geometry:
    DIM_EXTENSIBLE = True
    def __init__(self,dimension=2,values=None):
        self.dimension = dimension
        self.values = values
        if self.values is None:self.values = []
    def clone(self):
        return copy.deepcopy(self)
    def mbr(self):
        return None

EMPTY = Geometry()

class Rectangle(Geometry):
    def __init__(self,dimension=2,values=None):
        super(Rectangle,self).__init__(dimension,values)
    def boundary(self,dimension):
        if self.values is None or len(self.values)<=0:
            return None
        if dimension*2 > len(self.values):
            return None
        return [self.values[dimension*2],self.values[dimension*2+1]]
    def mbr(self):
        return self.clone()


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



