# 模块的导入
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

print(tf.__version__)

# 数据集导入及查看
imdb = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))

# 显示第一条影评
# 影评文本已转换为整数，其中每个整数都表示字典中的一个特定字词
print(train_data[0])

# 探索部分影评的长度
len(train_data[0]), len(train_data[1])

# 将整数转换回单词
# A dictionary mapping words to an integer index
word_index = imdb.get_word_index()

# The first indices are reserved
word_index = {k: (v+3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


# 显示第一条影评文本
decode_review(train_data[0])

# 使影评长度统一
# 为便于导入网络
train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)
# 进一步检验
len(train_data[0]), len(train_data[1])
print(train_data[0])  # 已填充后的

# 构建模型
# input shape is the vocabulary count used for the movie reviews (10,000 words)
# 输入形状是用于电影评论的词汇计数
vocab_size = 10000

# 第一层是 Embedding 层。该层会在整数编码的词汇表中查找每个字词-索引的嵌入向量。
# 模型在接受训练时会学习这些向量。这些向量会向输出数组添加一个维度。
# 生成的维度为：(batch, sequence, embedding)。
# 第二层三 GlobalAveragePooling1D 层通过对序列维度求平均值，
# 针对每个样本返回一个长度固定的输出向量。这样，模型便能够以尽可能简单的方式处理各种长度的输入。

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

model.summary()

# 配置模型以使用优化器和损失函数
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 创建验证集
x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)

# 模型的评估
results = model.evaluate(test_data, test_labels)
print(results)

# 创建损失率和损失岁时间变化的图
history_dict = history.history
history_dict.keys()

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

# 训练集与验证集上的损失
# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig("nlp_task/task_1/md_images/1-2.png")
plt.show()  # 由图可见，有较为明显的过拟合

# 清除画板，设置数据
plt.clf()   # clear figure
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']

# 训练集与验证集上的准确率
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig("nlp_task/task_1/md_images/1-3.png")
plt.show()

