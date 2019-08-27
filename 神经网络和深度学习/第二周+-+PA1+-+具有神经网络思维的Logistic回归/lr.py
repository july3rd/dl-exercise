from lr_utils import *
import matplotlib.pyplot as plt

# 加载数据集
train_set_x_orig,train_set_y_orig,test_set_x_orig,test_set_y_orig,classes = load_dataset()

# 将训练集进行降维并转置
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T

# 将测试集进行降维并转置
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T

# 数据归一化
train_set_x = train_set_x_flatten / 255
test_set_x = test_set_x_flatten / 255

# 开始训练
d = model(train_set_x,train_set_y_orig,test_set_x,test_set_y_orig,num_iter=2100,learning_rate=0.005,print_cost=True)

# 绘制学习曲线
costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations(per hundreds)')
plt.title("learning_rate = "+str(d['learning_rate']))
plt.show()

# 使用不同的学习率
learning_rates = [0.01,0.001,0.0001]
models = {}
for i in learning_rates:
    print("laearning rate is: "+str(i))
    models[str(i)] = model(train_set_x,train_set_y_orig,test_set_x,test_set_y_orig,num_iter=2000,learning_rate=i,print_cost=False)
    print('\n'+"------------------------------"+'\n')
    
for i in learning_rates:
    plt.plot(np.squeeze(models[str(i)]["costs"]),label = str(models[str(i)]["learning_rate"]))
    
plt.ylabel('cost')
plt.xlabel('iterations')

legend = plt.legend(loc='upper center',shadow = True)
frame = legend.get_frame()
frame.set_facecolor('0.90')
plt.show()
