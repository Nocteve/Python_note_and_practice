#文件读取：三个参数分别是文件路径(./data.txt)，
# 模式('r'为读取模式)，
# 编码模式(encoding='utf-8')
with open('./data.txt','r',encoding='utf-8') as f:
    text=f.read() #一次性读取全部内容，适用于小文件
    print(text)

with open('./data.txt','r',encoding='utf-8') as f:
    text_lines=f.readlines() #按行读取，适用于大文件
    for line in text_lines: #查看每一行
        print(line)
#注意文件读取时会把换行符\n也读取到，输出时体现为换行

#文件写入操作,'w'是覆盖式写入
with open('./data.txt','w',encoding='utf-8') as f:
    f.write('hello,world!\n') #换行需要手动添加\n
#'a'是追加式的写入
with open('./data.txt','a',encoding='utf-8') as f:
    f.writelines(['123123123\n','Nocteve ! ! !\n'])

#'r+'模式支持同时读写，但需注意文件指针位置（默认在文件开头，写入会覆盖指针位置的内容）