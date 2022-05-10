
import filecmp
import numpy as np

def dict_setattr(d:dict,keys:str,value):
    '''
    允许字典中
    :param d:
    :param keys:
    :param value:
    :return:
    '''
    strs = keys.split('.')
    if len(strs) == 1:
        d[keys] = value
    else:
        t = d
        for i in range(len(strs)-1):
            if keys[i] not in t.keys():
                t[keys[i]] = {}
            t = t[keys[i]]
        t[keys[-1]] = value
def dict_getattr(d:dict,keys:str):
    strs = keys.split('.')
    if len(strs) == 1:
        return dict[keys] if keys in dict.keys() else None
    else:
        t = d
        for i in range(len(strs) - 1):
            if keys[i] not in t.keys():
                return None
            t = t[keys[i]]
        return t[keys[-1]]

def set_dict_attr_function():
    dict.__setattr__ = dict_setattr
    dict.__getattr__ = dict_getattr

class Property:
    def __init__(self,name:str,desc:str,dvalue:str,optional:bool,type:str,constraints:str):
        self.name = name
        self.desc = desc
        self.dvalue = dvalue
        self.optional = optional
        self.type = type
        self.constraints = constraints
        set_dict_attr_function()
    def __str__(self):
        return self.name + ',' + self.desc + ',' + self.dvalue + ',' + self.optional + "," + self.type + "," +self.constraints
class Configuration:
    def __init__(self,filename:str='',filencode:str='utf-8',*props,**kwargs):
        self.filename = filename
        self.filencode = filencode
        self.props =props
        if self.props is None:self._props = []
        self.dict = {}
        self._load()

        if len(kwargs) > 0:
            for key,value in kwargs.items():
                self.dict[key] = value
    def serialize(self):
        return self.dict.copy()

    def __getitem__(self, item):
        return self.dict.get(item,None)
    def __getattr__(self, item):
        return self.dict.get(item,None)

    def _load(self):
        if self.filename is None or self.filename == '':
            return
        try:
            file = open(self.filename, 'r', encoding='utf-8')
            for line in file:
                if line.trim() == '':continue
                if line.trim().startsWith('#'):continue
                if line.find('=')<0:continue
                strs = line.split('=')
                self.dict.__setattr__(strs[0].trim(),strs[1].trim())
        except Exception as e:
            raise e


class Collections:
    @classmethod
    def all(cls,func,datas : list):
        if datas is None or len(datas)<=0:
            return True
        for d in datas:
            if not func(d):return False
        return True

    @classmethod
    def any(cls,func,datas:list):
        if datas is None or len(datas) <= 0:return False
        for d in datas:
            if func(d): return True
        return False

    @classmethod
    def map(cls,func,iterable,*params,**kwargs):
        for e in iterable:
            func(e,params,kwargs)
    @classmethod
    def assign(cls,prop,value,iterable):
        for e in iterable:
            e.__setattr__(prop,value)

    @classmethod
    def combinations(cls,source: list, n: int) -> list:
        '''从一个元素不重复的列表里选出n个元素
        :参数 source:元素不重复的列表
        :参数 n: 要选出的元素数量，正整数，小于等于列表source的长度
        '''
        # 如果n正好等于列表的长度，那么只有一种组合
        # 就是把所有元素都选出来
        if len(source) == n:
            return [source]
        # 如果n是1，那么列表中的每个元素都是一个组合
        if n == 1:
            ans = []
            for i in source:
                ans.append([i])
            return ans
        # 下面处理n小于列表长度的情况
        ans = []
        # 从列表里选n个元素出来，可以理解为先把列表里的第0个元素拿出来放进组合
        # 然后从剩下的元素里选出n-1个
        for each_list in Collections.combinations(source[1:], n - 1):
            ans.append([source[0]] + each_list)
        # 还可以直接从剩下的元素里选出n个
        for each_list in Collections.combinations(source[1:], n):
            ans.append(each_list)
        return ans

    @classmethod
    def group_cov(cls,g1,g2):
        """
        计算组间方差
        :param g1 list[float]
        :param g2 list[float]
        :
        """
        avg,avg1,avg2 = np.average(g1+g2),np.average(g1),np.average(g2)
        return (len(g1)*(avg-avg1)**2 + len(g2)*(avg-avg2)**2)/(len(g1)+len(g2))






