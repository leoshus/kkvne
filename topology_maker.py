import networkx as nx
import copy


def extract_network(path, filename):
    """read network and generate networkx.Graph instance """
    node_id, link_id = 0, 0
    with open(path + filename) as f:
        lines = f.readlines()
    if len(lines[0].split()) == 2:
        """create a substrate network"""
        node_num, link_num = [int(x) for x in lines[0].split()]
        graph = nx.Graph()
        build_node_edge(graph, lines, link_id, link_num, node_id, node_num)
    else:
        """create a virtual network"""
        node_num, link_num, time, duration, max_dis = [int(x) for x in lines[0].split()]
        graph = nx.Graph(type=0, time=time, duration=duration)
        build_node_edge(graph, lines, link_id, link_num, node_id, node_num)

    return graph


def build_node_edge(graph, lines, link_id, link_num, node_id, node_num):
    for line in lines[1:node_num + 1]:
        x, y, c = [float(x) for x in line.split()]
        graph.add_node(node_id, x_coordinate=x, y_coordinate=y, cpu=c, cpu_remain=c)
        node_id = node_id + 1
    for line in lines[-link_num:]:
        src, dst, bw, dis = [float(x) for x in line.split()]
        graph.add_edge(int(src), int(dst), link_id=link_id, bw=bw, bw_remain=bw, distance=dis)
        link_id = link_id + 1


def simulate_events_one(path, number):
    """读取number个虚拟网络，构成虚拟网络请求事件队列"""
    queue = []
    for i in range(number):
        filename = 'req%d.txt' % i
        req_arrive = extract_network(path, filename)
        req_arrive.graph['id'] = i
        req_leave = copy.deepcopy(req_arrive)
        req_leave.graph['type'] = 1
        req_leave.graph['time'] = req_arrive.graph['time'] + req_arrive.graph['duration']
        queue.append(req_arrive)
        queue.append(req_leave)
    # 按照时间（到达时间或离开时间）对这些虚拟网络请求从小到大进行排序
    queue.sort(key=lambda r: r.graph['time'])
    return queue
