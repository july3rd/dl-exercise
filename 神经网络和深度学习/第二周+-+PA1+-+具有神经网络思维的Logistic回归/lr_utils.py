import numpy as np
import h5py
    
    
def load_dataset():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes
	
	
	
def ini_param_with_zeros(dim):
    """
    将w初始化为维度为（dim，1）的0向量，将b初始化为set
    
    参数：
        dim - callable始化w的维度
        
    返回：
        w - 维度为（dim，1）的向量
        b - 初始化偏差0
    """
    w = np.zeros((dim,1))
    b = 0
    
    # 使用断言确保建立的是矩阵而不是数组
    assert(w.shape == (dim,1))
    assert(isinstance(b,float) or isinstance(b,int))
    
    return (w,b)
	
	
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
	

def propagate(w,b,X,Y):
    """
    实现前向和后向传播的成本函数及其梯度
    
    参数：
        w - 权重，维度为（64*64*3，1）的矩阵
        b - 偏差，标量
        X - 训练集，维度为（64*64*3，209）的矩阵
        Y - 标签，维度为（1，209）的矩阵
        
    返回：
        cost - 一轮迭代的代价
        dw - 权重的梯度
        db - 偏差的梯度 
    """
    
    # 训练集数量
    m = X.shape[1] 
    
    # 正向传播
    Z = np.dot(w.T,X) + b
    A = sigmoid(Z)
    cost = (-1 / m) * np.sum(Y * np.log(A) + (1-Y) * np.log(1 - A))
    
    # 反向传播
    dZ = A - Y 
    dw = (1 / m) * np.dot(X,dZ.T)
    db = (1 / m) * np.sum(dZ)
    
    # 使用断言确保数据是正确的
    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    
    # 创建一个字典，把dw和db存储起来
    grads = {
        "dw": dw,
        "db": db
    }
    return (grads,cost)


def optimize(w,b,X,Y,num_iter,learning_rate,print_cost=False):
    """
    多次迭代，找到全局最优的w和b
    
    参数：
        w - 权重
        b - 偏差
        X - 训练集
        Y - 标签
        num_iter - 迭代次数
        learning_rate - 学习率
        print_cost - 每迭代100次打印一次损失值
        
    返回：
        params - 
        grads -
        
    """
    costs = []
    
    for i in range(num_iter):
        
        grads,cost = propagate(w,b,X,Y)
        
        dw = grads["dw"]
        db = grads["db"]
        
        w = w - learning_rate * dw
        b = b - learning_rate * db
        
        # 记录代价
        if i % 100 == 0:
            costs.append(cost)
            
        # 打印成本数据
        if(print_cost) and (i % 100 == 0):
            print("迭代次数：%i，代价：%f" % (i,cost))
            
    params = {
        "w": w,
        "b": b
    }
    
    return (params,costs)
	
	
def predict(w,b,X):
    """
    使用习得的模型来预测训练集的标签是0或是1
    
    参数：
        w - 权重矩阵
        b - 偏差
        X - 训练集
        
    返回：
        Y_pre - 预测值
    """
    
    m = X.shape[1] # 训练集数量
    Y_pre = np.zeros((1,m))
    
    A = sigmoid(np.dot(w.T,X)+b)
    # 概率大于0.5，预测值为1，反之为0
    for i in range(A.shape[1]):
        Y_pre[0,i] = 1 if A[0,i] > 0.5 else 0
        
    assert(Y_pre.shape == (1,m))
    
    return Y_pre


def model(X_train,Y_train,X_test,Y_test,num_iter=2000,learning_rate = 0.5,print_cost=False):
    """
    应用之前实现的预测函数在训练集和测试集上预测标签
    
    参数：
        X_train - 训练集
        Y_train - 训练集标签
        X_test - 测试集
        Y_test - 测试集标签
        num_iter - 迭代次数
        learning_rate - 学习率
        print_cost - 设置true则每迭代100次打印一次代价
        
    返回：
        d - 
    """
    
    # 1.初始化参数
    w,b = ini_param_with_zeros(X_train.shape[0])
    
    # 2.优化参数（多次前向&后向传播）
    params,costs = optimize(w,b,X_train,Y_train,num_iter,learning_rate,print_cost)
    
    # 3.习得参数
    w,b = params["w"],params["b"]
    
    # 4.预测训练集/测试集
    Y_pre_train = predict(w,b,X_train)
    Y_pre_test = predict(w,b,X_test)

    # 打印训练得准确性
    print("训练集上得准确率：",format(100 - np.mean(np.abs(Y_pre_train-Y_train)) * 100),"%")
    print("测试集上得准确率：",format(100 - np.mean(np.abs(Y_pre_test-Y_test)) * 100),"%")
    
    d = {
        "costs": costs,
        "Y_pre_test": Y_pre_test,
        "Y_pre_train": Y_pre_train,
        "w": w,
        "b": b,
        "learning_rate": learning_rate,
        "num_iter": num_iter
    }
    
    return d
