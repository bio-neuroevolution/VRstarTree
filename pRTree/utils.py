
import filecmp

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
class Configuration:
    def __init__(self,filename:str,filencode:str='utf-8',*props):
        self.filename = filename
        self.filencode = filencode
        self.props =props
        if self.props is None:self._props = []
        self.dict = {}
        self._load()
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
