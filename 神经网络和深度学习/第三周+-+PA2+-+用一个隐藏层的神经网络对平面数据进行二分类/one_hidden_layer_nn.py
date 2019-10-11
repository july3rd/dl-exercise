import numpy as np
import matplotlib.pyplot as plt
from testCases import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary,sigmoid,load_planar_dataset,load_extra_datasets

np.random.seed(1)

def layer_sizes(X,Y):
    
    '''
    定义神经网络的结构
    参数：
    X - 输入数据集
    Y - 标签
    
    返回：
    n_x - 输入层单元个数
    n_h - 隐藏层单元个数
    n_y - 输出层单元个数
    '''
    n_x = X.shape[0] 
    n_h = 4
    n_y = Y.shape[0]
    
    return (n_x,n_h,n_y)


def initialize_parameters(n_x,n_h,n_y):
    """
    初始化参数
    
    参数：
        n_x - 输入层单元个数
        n_h - 隐藏层单元个数
        n_y - 输出层单元个数
        
    返回：
        W1 - 权重矩阵，维度为（n_h,n_x）
        b1 - 偏向量，维度为（n_h,1）
        
        W2 - 权重矩阵，维度为（n_y,n_h）
        b2 - 偏向量，维度为（n_y,1）
    """
    
    np.random.seed(2)
    W1 = np.random.randn(n_h,n_x) * 0.01 # 参数设置大的话，梯度下降会慢
    b1 = np.zeros(shape=(n_h,1))
    W2 = np.random.randn(n_y,n_h) * 0.01
    b2 = np.zeros(shape=(n_y,1))
    
    # 使用断言确保数据的维度是正确的
    assert(W1.shape == (n_h,n_x))
    assert(b1.shape == (n_h,1))
    assert(W2.shape == (n_y,n_h))
    assert(b2.shape == (n_y,1))
    
    parameters = {
        "W1":W1,
        "b1":b1,
        "W2":W2,
        "b2":b2
    }
    
    return parameters


def forward_propagation(X,parameters):
    """
    参数：
    X - 维度为（n_x,m）
    parameters - 参数
    
    返回：
    A2 - 使用sigmoid函数作为激活函数计算的输出矩阵
    cache - 包含"Z1","A1","Z2","A2"的字典类型变量
    """
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    # 前向传播计算A2
    Z1 = np.dot(W1,X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2,A1) + b2
    A2 = sigmoid(Z2)
    
    # 使用断言判断输出A1的维度
    assert(A2.shape == (1,X.shape[1]))
    
    cache = {
        "Z1":Z1,
        "A1": A1,
        "Z2": Z2,
        "A2": A2
    }
    
    return (A2,cache)


def compute_cost(A2,Y):
    """
    计算交叉熵成本
    
    参数：
    A2 - 使用sigmoid激活函数计算出的预测值向量
    Y - 标签数据
    
    返回：
    cost - 成本
    """
    
    m = Y.shape[1]
    
    logprobs = np.multiply(np.log(A2),Y)  + np.multiply(np.log(1-A2),(1-Y))
    cost = -np.sum(logprobs) / m
    cost = float(np.squeeze(cost))
    
    assert(isinstance(cost,float))
    
    return cost

def backward_propagation(parameters,cache,X,Y):
    """
    反向传播函数
    
    参数：
        parameters - 包含权重向量和偏置向量的参数字典
        cache - 包含“Z1”,"A1","Z2","A2"的参数字典
        X - 输入数据
        Y - 标签
        
    返回：
        grads - 包含W和b的倒数的字典变量
    """
    
    m = X.shape[1]
    
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    
    A1 = cache["A1"]
    A2 = cache["A2"]
    
    dZ2 = A2 - Y
    dW2 = (1 / m) * np.dot(dZ2,A1.T)
    db2 = (1 / m) * np.sum(dZ2,axis=1,keepdims=True)
    dZ1 = np.multiply(np.dot(W2.T,dZ2),1-np.power(A1,2))
    dW1 = (1 / m) * np.dot(dZ1,X.T)
    db1 = (1 / m) * np.sum(dZ1,axis=1,keepdims=True)
    
    grads = {
        "dW1":dW1,
        "db1":db1,
        "dW2":dW2,
        "db2":db2
    }
    
    return grads

def update_parameters(parameters,grads,learning_rate=1.2):
    """
    使用上面给出的梯度下降更新规则更新参数

    参数：
     parameters - 包含参数的字典类型的变量。
     grads - 包含导数值的字典类型的变量。
     learning_rate - 学习速率

    返回：
     parameters - 包含更新参数的字典类型的变量。
    """
    W1,W2 = parameters["W1"],parameters["W2"]
    b1,b2 = parameters["b1"],parameters["b2"]

    dW1,dW2 = grads["dW1"],grads["dW2"]
    db1,db2 = grads["db1"],grads["db2"]

    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters

def nn_model(X,Y,n_h,num_iterations,print_cost=False):
    """
    参数：
        X - 数据集
        Y - 标签
        n_h - 隐藏层单元个数
        num_iterationsum - 梯度下降迭代次数
        print_cost - 如果设置为True，则每1000次迭代打印一次成本值
        
    返回：
        parameters - 模型学得的参数
    """
    
    np.random.seed(3)
    n_x = layer_sizes(X,Y)[0]
    n_y = layer_sizes(X,Y)[2]
    
    parameters = initialize_parameters(n_x,n_h,n_y)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    for i in range(num_iterations):
        A2,cache = forward_propagation(X,parameters)
        cost = compute_cost(A2,Y)
        grads = backward_propagation(parameters,cache,X,Y)
        parameters = update_parameters(parameters,grads,learning_rate=1.2)
        
        if i % 1000 == 0 and print_cost:
            print("在第",i,"次迭代后，代价为："+str(cost))
            
        
#         print("在第",i,"次迭代后，代价为："+str(cost))
            
    return parameters

def predict(parameters,X):
    """
    使用学习的参数，为X中的每一个示例预测一个分类
    
    参数：
        parameters - 参数
        X - 输入数据
        
    返回：
        predictions - 预测值
"""
    A2,cache = forward_propagation(X,parameters)
    predictions = np.round(A2)
    
    return predictions



# 加载数据集
X,Y = load_planar_dataset()

# 训练参数
parameters = nn_model(X,Y,n_h=4,num_iterations=10000,print_cost=True)

# 开始预测
predictions = predict(parameters,X)

# 打印准确率
print('准确率：%d' % float((np.dot(Y,predictions.T)+np.dot(1-Y,1-predictions.T)) / float(Y.size)*100) + '%')