# 模块导入
import os
from collections import Counter
import numpy as np
import tensorflow.keras as kr

# 文件路径设置
os.chdir('task_1')
base_dir = 'data/cnews'
train_dir = os.path.join(base_dir, 'cnews.train.txt')
test_dir = os.path.join(base_dir, 'cnews.test.txt')
val_dir = os.path.join(base_dir, 'cnews.val.txt')
vocab_dir = os.path.join(base_dir, 'cnews.vocab.txt')


# 功能函数
def open_file(filename, mode='r'):
    """
    mode: 'r' or 'w' for read or write
    """
    return open(filename, mode)


def read_file(filename):
    """读取文件数据"""
    contents, labels = [], []
    with open_file(filename) as f:
        for line in f:
            try:
                label, content = line.strip().split('\t')
                if content:
                    contents.append(list(content))
                    labels.append(label)
            except:
                pass
    return contents, labels


def build_vocab(train_dir, vocab_dir, vocab_size=5000):
    """根据训练集构建词汇表，存储"""
    data_train, _ = read_file(train_dir)

    all_data = []
    for content in data_train:
        all_data.extend(content)

    counter = Counter(all_data)
    count_pairs = counter.most_common(vocab_size - 1)
    words, _ = list(zip(*count_pairs))
    # 添加一个 <PAD> 来将所有文本pad为同一长度
    words = ['<PAD>'] + list(words)
    open_file(vocab_dir, mode='w').write('\n'.join(words) + '\n')


def read_vocab(vocab_dir):
    """读取词汇表"""
    # words = open_file(vocab_dir).read().strip().split('\n')
    with open_file(vocab_dir) as fp:
        # 如果是py2 则每个值都转化为unicode
        words = [_.strip() for _ in fp.readlines()]
    word_to_id = dict(zip(words, range(len(words))))
    return words, word_to_id


def read_category():
    """读取分类目录，固定"""
    categories = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']
    # categories = [x for x in categories]
    cat_to_id = dict(zip(categories, range(len(categories))))

    return categories, cat_to_id


def to_words(content, words):
    """将id表示的内容转换为文字"""
    return ''.join(words[x] for x in content)


def process_file(filename, word_to_id, cat_to_id, max_length=600):
    """将文件转换为id表示"""
    contents, labels = read_file(filename)

    data_id, label_id = [], []
    for i in range(len(contents)):
        data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
        label_id.append(cat_to_id[labels[i]])

    # 使用keras提供的pad_sequences来将文本pad为固定长度
    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length)
    y_pad = kr.utils.to_categorical(label_id, num_classes=len(cat_to_id))  # 将标签转换为one-hot表示

    return x_pad, y_pad


def batch_iter(x, y, batch_size=64):
    """生成批次数据"""
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1

    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x[indices]
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]


# 数据探索
# 读取文本，查看文本信息
x_ori, y_ori = read_file(train_dir)
print(len(x_ori), len(y_ori))
x_val_ori, y_val_ori = read_file(val_dir)
print(len(x_val_ori), len(y_val_ori))

# 构建词汇表(数据集中已有）
build_vocab(train_dir, vocab_dir, vocab_size=5000)

# 读取分类目录
categories, cat_to_id = read_category()
print(categories, cat_to_id)

# 输出类的个数
# 训练集
for i in categories:
    print(str(i) + ":" + str(y_ori.count(i)))
# 验证集
for i in categories:
    print(str(i) + ":" + str(y_val_ori.count(i)))

# 获取词汇表
words, word_to_id = read_vocab(vocab_dir)

# 建立训练集和测试集
# 将文本转换为id的形式, 便于后续放入网络中
x_train, y_train = process_file(train_dir, word_to_id, cat_to_id, max_length=600)
x_val, y_val = process_file(val_dir, word_to_id, cat_to_id, max_length=600)
print(x_train[0])
print(y_train[0])
print([len(x_train[i]) for i in range(3)])  # 查看单句长度

# 将id的形式转换为文本
print(to_words(x_train[0], words))
