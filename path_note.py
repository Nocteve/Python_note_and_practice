#一些文件路径的写法，在构建数据集的时候很有用
#当然直接使用字符串也可以，但是使用现代化的写法更安全先进
from pathlib import Path  # pathlib库是Python3.4+引入的**面向对象**的文件系统路径处理库，提供了更直观、易用的 API
import os # os库用于与操作系统交互，提供一系列有关于文件目录的函数
import time

def About_os():
    # 路径拼接
    path = os.path.join('folder', 'subfolder', 'file.txt')
    print(f'Path:{path}')
    # 输出: folder/subfolder/file.txt (Windows: folder\subfolder\file.txt)

    # 路径分割
    dirname, filename = os.path.split('/home/user/file.txt')
    print(f'dirname:{dirname} filename:{filename}')
    # dirname = '/home/user', filename = 'file.txt'

    basename = os.path.basename('/home/user/file.txt')  # 'file.txt'
    dirname = os.path.dirname('/home/user/file.txt')    # '/home/user'
    print(f'basename:{basename} dirname:{dirname}')

    # 获取扩展名
    root, ext = os.path.splitext('/home/user/file.txt')  
    print(f'root:{root} ext(extension):{ext}')
    # root = '/home/user/file', ext = '.txt'

    # 规范化路径
    normalized = os.path.normpath('/home//user/../user/file.txt')
    print(f'normalized:{normalized}')
    # '/home/user/file.txt'

    # 绝对路径
    abs_path = os.path.abspath('file.txt')
    print(f'abs_path:{abs_path}')

    files = os.listdir('.')  # 当前目录所有内容
    print(f'files in this dir:{files}')

    # 检查路径类型
    print(os.path.exists('/path/to/file'))  # 是否存在
    print(os.path.isfile('/path/to/file'))  # 是否是文件
    print(os.path.isdir('/path/to/dir'))    # 是否是目录
    print(os.path.islink('/path/to/link'))  # 是否是链接

    # 获取文件信息
    scipt_dir=os.path.dirname(os.path.abspath(__file__))#获取当前文件所在路径
    print(scipt_dir)
    size = os.path.getsize(os.path.join(scipt_dir,'data.txt'))  # 文件大小（字节）
    mtime = os.path.getmtime(os.path.join(scipt_dir,'data.txt'))  # 最后修改时间（时间戳）
    atime = os.path.getatime(os.path.join(scipt_dir,'data.txt'))  # 最后访问时间
    ctime = os.path.getctime(os.path.join(scipt_dir,'data.txt'))  # 创建时间/元数据修改时间
    #ps. 这里注意一下，如果直接写data.txt，比如os.path.getsize('data.txt'),是相对于运行脚本的相对路径，而非该文件的相对路径

    print(f'size:{size}')
    # 转换为可读时间
    print(time.ctime(mtime)) #time.ctime函数用于将时间戳（以秒为单位的浮点数）转换为易于阅读的字符串形式

def About_pathlib():
    # 创建 Path 对象
    p = Path('.')  # 当前目录 (运行脚本的目录）
    p = Path('/home/user')  # 绝对路径
    p = Path('dir/subdir')  # 相对路径

    # 跨平台路径拼接（使用 / 运算符）
    config_path = Path.home() / '.config' / 'myapp' / 'config.json'
    print(Path.home())
    print(Path(__file__))
    path = Path('/home/user/docs/file.txt')

    # 获取路径各部分
    print(path.parent)      # /home/user/docs
    print(path.name)        # file.txt
    print(path.stem)        # file (无扩展名)
    print(path.suffix)      # .txt
    print(path.suffixes)    # ['.txt'] (对于.tar.gz返回['.tar', '.gz'])
    print(path.parts)       # ('/', 'home', 'user', 'docs', 'file.txt')

    # 路径组合
    new_path = path.parent / 'newfile.txt'
    print(new_path)

    # 目录遍历与文件搜索
    #列出目录所有内容
    p=Path(__file__).parent
    for child in p.iterdir():# iterator(迭代器) dir(目录)
        print(child.name)

    # 递归遍历所有文件(包括子目录里的)
    for file_path in p.rglob('*.py'):  # 递归查找所有.py文件 Recursion(递归)
        print(file_path.name)

    # 使用glob模式匹配
    for py_file in p.glob('**/*.py'):  # 等同于rglob (**代表递归所有目录)
        print(py_file.name)

    # 筛选文件
    txt_files = list(p.glob('*.txt'))
    subdirs = [x for x in p.iterdir() if x.is_dir()]


    p = Path(Path(__file__).parent / 'data.txt')
    # 路径类型判断
    print(p.exists())    # 是否存在
    print(p.is_file())   # 是否是文件
    print(p.is_dir())    # 是否是目录
    print(p.is_symlink())  # 是否是符号链接
    print(p.is_absolute())  # 是否是绝对路径

    # 获取统计信息
    stat = p.stat()
    print(f"大小: {stat.st_size} 字节")
    print(f"修改时间: {time.ctime(stat.st_mtime)}")
    print(f"权限: {oct(stat.st_mode)}")

    # 解析路径
    print(p.resolve())  # 解析符号链接，返回绝对路径
    print(p.absolute())  # 返回绝对路径

    # 常用路径
    home = Path.home()  # 用户家目录
    cwd = Path.cwd()    # 当前工作目录(运行脚本的路径，不是当前文件的路径)
    # 构建路径
    config_dir = Path.home() / '.config' / 'myapp'
    temp_file = Path('/tmp') / 'temp.txt'

if __name__=='__main__':
    print('About os:')
    About_os()
    print('\nAbout pathlib')
    About_pathlib()