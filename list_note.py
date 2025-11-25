l=[] #创建一个空列表
l1=[1,2,'a','hello',l] #创建一个带初始化元素的列表

print(f'list l1:{l1}')
print(f'l1[0]:{l1[0]}')
print(f'l1[1:5]:{l1[1:5]}') #列表切片，左闭右开
'''
list l1:[1, 2, 'a', 'hello', []]
l1[0]:1
l1[1:5]:[2, 'a', 'hello', []]
'''

l.append('hello') #向列表中添加元素
l.append(1)
l.append(2)
l.append('a')

print('\n')

print(f'origin:{l}') #删除列表中的元素
del l[1]
print(f'after del:{l}')
'''
origin:['hello', 1, 2, 'a']
after del:['hello', 2, 'a']
'''

print('\n')

print(len(l)) #获取列表长度
l=l*2 #重复列表
print(l)
del l1[4]
l=l+l1 #列表拼接
print(l)
for x in l: #列表迭代
    print(x)

l=l[2:] #从索引2处截取列表
print(l)
l.extend(['a','b']) #追加多个元素
print(l)
print(f'count a in l: {l.count('a')}') #统计某个元素的个数

l2=[1,2,4,6,0,12,12,2]
print(f'max in l2:{max(l2)}')#最大值
print(f'min in l2:{min(l2)}')#最小值
l2.sort(reverse=True)#排序,默认从小到大，reverse=True时从大到小
print(l2)

