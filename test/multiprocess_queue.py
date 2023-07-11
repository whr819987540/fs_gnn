from multiprocessing import Process, Queue

# 定义一个子进程函数
def worker(queue):
    # 从队列中获取数据
    data = queue.get()
    print("子进程收到数据:", data)

# 创建一个队列
queue = Queue()

# 创建一个子进程，并传递队列作为参数
p = Process(target=worker, args=(queue,))

# 启动子进程
p.start()

# 向队列中放入数据
data = {"name": "python", "age": 20, "addr": "beijing"}
queue.put(data)

# 等待子进程结束
p.join()

print("主进程结束")
