from functools import reduce
import random


def topologic(graph):
    sorted_node = []

    while graph:
        all_nodes_have_inputs = reduce(lambda a, b: a + b, list(graph.values()))
        all_nodes_have_outputs = list(graph.keys())

        all_node_only_have_outputs_no_inputs = list(set(all_nodes_have_outputs) - set(all_nodes_have_inputs))

        if all_node_only_have_outputs_no_inputs:
            node = random.choice(all_node_only_have_outputs_no_inputs)

            sorted_node.append(node)

            if len(graph) == 1:
                sorted_node += graph[node]

            graph.pop(node)

            for _, links in graph.items():
                if node in links: links.remove(node)
        else:
            raise TypeError('this graph have circle, which canot get topologic order')

    return sorted_node


x, k, b, linear, sigmoid, y, loss = 'x', 'k', 'b', 'linear', 'sigmoid', 'y', 'loss'
test_graph = {
    x: [linear],
    k: [linear],
    b: [linear],
    linear: [sigmoid],
    sigmoid: [loss],
    y: [loss]
}

sorted_order = topologic(test_graph)
print(sorted_order)

import numpy as np


class Node():
    def __init__(self, inputs=[], name=None, is_trainable=False):
        self.inputs = inputs
        self.outputs = []
        self.name = name
        self.value = None
        self.gradients = dict()
        self.is_trainable = is_trainable

        for node in inputs:
            node.outputs.append(self)

    def __repr__(self):
        return 'Node: {}'.format(self.name)

    def forward(self):
        print('I am {}, i have no human baba,i caculate myself value by myself !!!'.format(self.name))

    def backward(self):
        pass


class Placeholder(Node):
    def __init__(self, name=None, is_trainable=False):
        Node.__init__(self, name=name, is_trainable=is_trainable)

    def __repr__(self):
        return 'Placeholder: {}'.format(self.name)

    def forward(self):
        print('I am {}, i am assigned value {} by human baba'.format(self.name, self.value))

    def backward(self):
        self.gradients[self] = self.outputs[0].gradients[self]


class Linear(Node):
    def __init__(self, x, k, b, name=None):
        Node.__init__(self, inputs=[x, k, b], name=name)  # super(Node,self).__init__(inputs=[x,k,b],name=name)

    def __repr__(self):
        return 'Linear: {}'.format(self.name)

    def forward(self):
        x, k, b = self.inputs[0], self.inputs[1], self.inputs[1]
        self.value = x.value * k.value + b.value
        print('I am {}, i have no human baba,i caculate myself value {} by myself !!!'.format(self.name, self.value))

    def backward(self):
        x, k, b = self.inputs[0], self.inputs[1], self.inputs[1]
        self.gradients[self.inputs[0]] = self.outputs[0].gradients[self] * k.value
        self.gradients[self.inputs[1]] = self.outputs[0].gradients[self] * x.value
        self.gradients[self.inputs[2]] = self.outputs[0].gradients[self] * 1
        #         self.gradients[self.inputs[0]] = '*'.join([self.outputs[0].gradients[self],'∂{}/∂{}'.format(self.name,self.inputs[0].name)])
        #         self.gradients[self.inputs[1]] = '*'.join([self.outputs[0].gradients[self],'∂{}/∂{}'.format(self.name,self.inputs[1].name)])
        #         self.gradients[self.inputs[2]] = '*'.join([self.outputs[0].gradients[self],'∂{}/∂{}'.format(self.name,self.inputs[2].name)])
        print('self.gradients[self.inputs[0]] {}'.format(self.gradients[self.inputs[0]]))
        print('self.gradients[self.inputs[1]] {}'.format(self.gradients[self.inputs[1]]))
        print('self.gradients[self.inputs[2]] {}'.format(self.gradients[self.inputs[2]]))


class Sigmoid(Node):
    def __init__(self, x, name=None):
        Node.__init__(self, inputs=[x], name=name)

    def __repr__(self):
        return 'Linear: {}'.format(self.name)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self):
        x = self.inputs[0]
        self.value = self._sigmoid(x.value)
        print('I am {}, i have no human baba,i caculate myself value {} by myself !!!'.format(self.name, self.value))

    def backward(self):
        x = self.inputs[0]
        self.gradients[self.inputs[0]] = self.outputs[0].gradients[self] * self._sigmoid(x.value) * (
                    1 - self._sigmoid(x.value))
        #         self.gradients[self.inputs[0]] = '*'.join([self.outputs[0].gradients[self],'∂{}/∂{}'.format(self.name,self.inputs[0].name)])
        print('self.gradients[self.inputs[0]] {}'.format(self.gradients[self.inputs[0]]))


class Loss(Node):
    def __init__(self, y, y_hat, name=None):
        Node.__init__(self, inputs=[y, y_hat], name=name)

    def __repr__(self):
        return 'Loss: {}'.format(self.name)

    def forward(self):
        y, y_hat = self.inputs[0], self.inputs[1]
        self.value = np.mean((y.value - y_hat.value) ** 2)
        print('I am {}, i have no human baba,i caculate myself value {} by myself !!!'.format(self.name, self.value))

    def backward(self):
        y, y_hat = self.inputs[0], self.inputs[1]
        self.gradients[self.inputs[0]] = 2 * np.mean(y.value - y_hat.value)
        self.gradients[self.inputs[1]] = -2 * np.mean(y.value - y_hat.value)
        #         self.gradients[self.inputs[0]] = '∂{}/∂{}'.format(self.name,self.inputs[0].name)
        #         self.gradients[self.inputs[1]] = '∂{}/∂{}'.format(self.name,self.inputs[1].name)
        print('self.gradients[self.inputs[0]] {}'.format(self.gradients[self.inputs[0]]))
        print('self.gradients[self.inputs[1]] {}'.format(self.gradients[self.inputs[1]]))


node_x = Placeholder(name='x')
node_k = Placeholder(name='k', is_trainable=True)
node_b = Placeholder(name='b', is_trainable=True)
node_y = Placeholder(name='y')
node_linear = Linear(node_x, node_k, node_b, name='linear')
node_sigmoid = Sigmoid(x=node_linear, name='sigmoid')
node_loss = Loss(y_hat=node_sigmoid, y=node_y, name='loss')

feed_dict = {
    node_x: 3,
    node_k: random.random(),
    node_b: 0.38,
    node_y: random.random()
}

from collections import defaultdict


def convert_feed_dict_to_graph(feed_dict):
    computing_graph = defaultdict(list)  # 构建一个默认value为list的字典
    need_expand = [n for n in feed_dict]
    while need_expand:
        n = need_expand.pop(0)
        if n in computing_graph: continue
        if isinstance(n, Placeholder): n.value = feed_dict[n]
        for m in n.outputs:
            computing_graph[n].append(m)
            need_expand.append(m)
    return computing_graph


sorted_nodes = topologic(convert_feed_dict_to_graph(feed_dict))
print(sorted_nodes)


def forward(graph_sorted_nodes):
    for node in graph_sorted_nodes:
        node.forward()


def backward(graph_sorted_nodes):
    for node in sorted_nodes[::-1]:
        #         print('\nI am {}'.format(node.name))
        node.backward()


def run_one_epoch(graph_sorted_nodes):
    forward(graph_sorted_nodes)
    backward(graph_sorted_nodes)


def optimize(graph_notes, learning_rate=1e-1):
    for node in graph_notes:
        if node.is_trainable:
            node.value = node.value + -1 * node.gradients[node] * learning_rate


loss_history = []
for _ in range(100):
    run_one_epoch(sorted_nodes)
    _loss_node = sorted_nodes[-1]
    assert (_loss_node, Loss)
    loss_history.append(_loss_node.value)
    optimize(sorted_nodes)

import matplotlib.pyplot as plt

plt.plot(loss_history)   