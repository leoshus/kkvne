from topology_maker import extract_network
import copy
import tensorflow.compat.v1 as tf
from RLN import RLN


def configure(sub, name, arg):
    training_set_path = 'kk/training_set/'
    training_set = simulate_events_one(training_set_path, 1000)
    # RLN构造器
    rln = RLN(sub=sub,
              n_actions=sub.net.number_of_nodes(),
              n_features=6,  # 节点向量维度 衡量节点映射好坏的维度
              learning_rate=0.05,  # 学习率
              num_epoch=arg,  # 训练轮次
              batch_size=100)  # 批量运算多少次后更新模型
    # 根据训练集训练模型
    rln.train(training_set)
    nodeSaver = tf.train.Saver()
    nodeSaver.save(rln.sess, "./kk/node_model/node_model.ckpt")
    return rln


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
    # 按照时间(到达事件或离开时间)对这些虚拟网络请求从小到大进行排序
    queue.sort(key=lambda r: r.graph['time'])
    return queue
