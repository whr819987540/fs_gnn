import multiprocessing
import threading
import signal
import os
import time

def child_thread_func():
    signal.pause()  # 子线程暂停，等待信号

def child_process_func():
    print("子进程启动")
    time.sleep(5)  # 模拟子进程的执行时间
    print("子进程结束")
    os.kill(os.getpid(), signal.SIGUSR1)  # 发送自定义信号给主进程

def main_process():
    signal.signal(signal.SIGUSR1,)
    print("主进程启动")
    child_thread = threading.Thread(target=child_process_func)
    child_thread.start()
    
    child_process = multiprocessing.Process(target=child_process_func)
    child_process.start()
    child_process.join()  # 等待子进程结束
    
    print("主进程收到子进程结束信号，通知子线程退出")
    os.kill(child_thread.ident, signal.SIGUSR1)  # 发送自定义信号给子线程
    child_thread.join()
    
    print("主进程退出")

if __name__ == "__main__":
    main_process()
