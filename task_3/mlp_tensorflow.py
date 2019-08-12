import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 导入MNIST手写数据# 
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

# 注册默认的session，之后的运算都会在这个session中进行# 
sess = tf.InteractiveSession()

# 定义输入层神经元个数# 
in_units = 784

# 定义隐层神经元个数# 
h1_units = 512
h2_units = 512

# 为输入层与隐层神经元之间的连接权重初始化持久的正态分布随机数，这里权重为784乘300，300是隐层的尺寸# 
W1 = tf.Variable(tf.truncated_normal([in_units, h1_units], mean=0, stddev=0.2))
b1 = tf.Variable(tf.zeros([h1_units]))

W2 = tf.Variable(tf.truncated_normal([h1_units, h2_units]))
b2 = tf.Variable(tf.zeros([h2_units]))

W3 = tf.Variable(tf.zeros([h2_units, 10]))
b3 = tf.Variable(tf.zeros([10]))

# 定义自变量的输入部件，尺寸为 任意行X784列# 
x = tf.placeholder(tf.float32, [None, in_units])

# 为dropout中的保留比例设置输入部件# 
keep_prob = tf.placeholder(tf.float32)


hidden1 = tf.nn.sigmoid(tf.matmul(x, W1) + b1)  # 定义隐层求解部件
hidden1_drop = tf.nn.dropout(hidden1, keep_prob)  # 定义隐层dropout操作部件

hidden2 = tf.nn.sigmoid(tf.matmul(hidden1_drop, W2) + b2)  # 定义隐层求解部件
hidden2_drop = tf.nn.dropout(hidden2, keep_prob)  # 定义隐层dropout操作部件

y = tf.nn.sigmoid(tf.matmul(hidden2_drop, W3) + b3)  # 定义输出层sigmoid计算部件


y_ = tf.placeholder(tf.float32, [None, 10])  # 定义训练label的输入部件

# 定义均方误差计算部件，这里注意要压成1维
loss_function = tf.reduce_mean(tf.reduce_sum((y_ - y)**2, reduction_indices=[1]))

# 定义优化器组件，这里采用AdagradOptimizer作为优化算法，这是种变种的随机梯度下降算法
train_step = tf.train.AdagradOptimizer(0.18).minimize(loss_function)

# 激活当前session中的全部部件# 
tf.global_variables_initializer().run()

# 开始迭代训练过程，最大迭代次数为3001次# 
for i in range(15000):
    # 为每一轮训练选择一个尺寸为100的随机训练批# 
    batch_xs, batch_ys = mnist.train.next_batch(100)
    # 将当前轮迭代选择的训练批作为输入数据输入train_step中进行训练# 
    train_step.run({x: batch_xs, y_: batch_ys, keep_prob: 0.76})
    # 每500轮打印一次当前网络在测试集上的训练结果# 
    if i % 50 == 0:
        print('第', i, '轮迭代后：')
        # 构造bool型变量用于判断所有测试样本与其真是类别的匹配情况# 
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        # 将bool型变量转换为float型并计算均值# 
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # 激活accuracy计算组件并传入mnist的测试集自变量、标签及dropout保留比率，这里因为是预测所以设置为全部保留# 
        print(accuracy.eval({x: mnist.test.images,
                             y_: mnist.test.labels,
                             keep_prob: 1.0}))