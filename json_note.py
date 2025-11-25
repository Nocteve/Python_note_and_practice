import json 
import numpy

'''
json -> python 
python -> json 
'''
'''
json  <--->  python 
对象{}       字典(dict)
数组[]       列表(list)

字符串方面json仅支持双引号
bool值方面,json首字母小写(true/false),python首字母大写(True/False)
'''
'''
json模块四个核心函数
json.dumps()  Python 对象 → JSON 字符串（序列化）
json.loads()  JSON 字符串 → Python 对象（反序列化）
json.dump()  Python 对象 → JSON 文件（序列化 + 写文件）
json.load()  JSON 文件 → Python 对象（读文件 + 反序列化）
'''
d={'name':'nocteve','id':1,'a':None,'b':'True','c':[1,2,3]}
print(d)
json_str=json.dumps(
    d,
    indent=4,          # 缩进4个空格
    ensure_ascii=False, # 不编码中文
    sort_keys=True      # 按键排序
    )
print(json_str)
l=[1,2,3]
json_str=json.dumps(l)
print(json_str)

json_str='''
{
    "a": null,
    "b": "True",
    "c": [
        1,
        2,
        3
    ],
    "id": 1,
    "name": "nocteve"
}
'''
d=json.loads(json_str)
print(d['name'])
