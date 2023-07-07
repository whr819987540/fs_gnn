import logging
import multiprocessing
import threading
import signal

def worker(queue,counter):
    # logger = multiprocessing.get_logger()
    # logger.addHandler(logging.StreamHandler())  # 添加一个处理器来将日志消息输出到标准输出流
    # logger.setLevel(logging.INFO)
    # logger.info("This is a log message from process {}".format(multiprocessing.current_process().name))
    
    # 将日志消息放入队列中，以便其他进程处理
    queue.put("Log message from process {}".format(multiprocessing.current_process().name))
    with counter.get_lock():
        counter.value -= 1

def main():
    counter = multiprocessing.Value("i",5)
    # 创建日志队列
    log_queue = multiprocessing.Queue()
    
    # 创建日志处理器，用于从队列中接收日志消息并输出到标准输出流
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(processName)s - %(message)s'))
    # 创建日志记录器，并将处理器添加到记录器中
    logger = multiprocessing.get_logger()
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    
    # 创建多个工作进程
    processes = []
    for i in range(5):
        p = multiprocessing.Process(target=worker, args=(log_queue,counter,))
        processes.append(p)
        p.start()

    output_thread = threading.Thread(target=output,args=(log_queue,logger,counter))
    output_thread.start()
    
    # 等待所有进程完成
    for p in processes:
        p.join()

def output(log_queue,logger,counter):
    # 从队列中获取其他进程放入的日志消息，并进行处理
    while True:
        with counter.get_lock():
            if counter.value == 0:
                logger.info("over")
                break
        message = log_queue.get()
        logger.info(message)   

if __name__ == "__main__":
    main()
