import psutil

def get_network_usage(interface):
    net_io = psutil.net_io_counters(pernic=True)
    if interface in net_io:
        io = net_io[interface]
        recv_bytes = io.bytes_recv
        sent_bytes = io.bytes_sent
        return recv_bytes, sent_bytes
    else:
        return None

interface = 'lo'  # 替换为你要获取的网卡名称
usage = get_network_usage(interface)
if usage:
    recv_bytes, sent_bytes = usage
    print(f"接收量：{recv_bytes} 字节")
    print(f"发送量：{sent_bytes} 字节")
else:
    print("网卡不存在")
