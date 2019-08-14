# coding=utf-8
import tensorflow as tf
from datetime import datetime
import os
from load_data import load_dataset, load_dataset_from_pickle
from textcnn_case_2_model import TextCNN
from textcnn_case_2_model import Settings

# Data loading params
tf.flags.DEFINE_string("train_data_path", 'data/train_query_pair_test_data.pickle', "data directory")
tf.flags.DEFINE_string("embedding_W_path", "./data/embedding_matrix.pickle", "pre-trained embedding matrix")
tf.flags.DEFINE_integer("vocab_size", 3627705, "vocabulary size")  # **这里需要根据词典的大小设置**
tf.flags.DEFINE_integer("num_classes", 2, "number of classes")
tf.flags.DEFINE_integer("embedding_size", 100, "Dimensionality of character embedding (default: 200)")
tf.flags.DEFINE_integer("batch_size", 256, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 1, "Number of training epochs (default: 50)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_integer("max_words_in_doc", 30, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_integer("evaluate_every", 100, "evaluate every this many batches")
tf.flags.DEFINE_float("learning_rate", 0.001, "learning rate")
tf.flags.DEFINE_float("keep_prob", 0.5, "dropout rate")

FLAGS = tf.flags.FLAGS

train_x, train_y, dev_x, dev_y, W_embedding = load_dataset_from_pickle(FLAGS.train_data_path, FLAGS.embedding_W_path)
train_sample_n = len(train_y)
print(len(train_y))
print(len(dev_y))
print("data load finished")
print("W_embedding : ", W_embedding.shape[0], W_embedding.shape[1])

# 模型的参数配置
settings = Settings(()
"""
可以配置不同的参数,需要根据训练数据集设置 vocab_size embedding_size
"""
settings.embedding_size = FLAGS.embedding_size
settings.vocab_size = FLAGS.vocab_size

# 设置GPU的使用率
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

with tf.Session() as sess:
# 在session中, 首先初始化定义好的model
    textcnn = TextCNN(settings=settings, pre_trained_word_vectors=W_embedding)

# 在train.py 文件中定义loss和accuracy, 这两个指标不要再model中定义
with tf.name_scope('loss'):
# print textcnn._inputs_y
# print textcnn.predictions
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=textcnn.scores,
                                                                  labels=textcnn._inputs_y,
                                                                  name='loss'))
with tf.name_scope('accuracy'):
# predict = tf.argmax(textcnn.predictions, axis=0, name='predict')
    predict = textcnn.predictions  # 在模型的定义中, textcnn.predictions 已经是经过argmax后的结果, 在训练.py文件中不能再做一次argmax
label = tf.argmax(textcnn._inputs_y, axis=1, name='label')
# print predict.get_shape()
# print label.get_shape()
acc = tf.reduce_mean(tf.cast(tf.equal(predict, label), tf.float32))

# make一个文件夹, 存放模型训练的中间结果
timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
timestamp = "textcnn" + timestamp
out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
print("Writing to {}\n".format(out_dir))

# 定义一个全局变量, 存放到目前为止,模型优化迭代的次数
global_step = tf.Variable(0, trainable=False)

# 定义优化器, 找出需要优化的变量以及求出这些变量的梯度
optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
tvars = tf.trainable_variables()
grads = tf.gradients(loss, tvars)
grads_and_vars = tuple(zip(grads, tvars))
train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)  # 我理解, global_step应该会在这个函数中自动+1

# 不优化预训练好的词向量
tvars_no_embedding = [tvar for tvar in tvars if 'embedding' not in tvar.name]
grads_no_embedding = tf.gradients(loss, tvars_no_embedding)
grads_and_vars_no_embedding = tuple(zip(grads_no_embedding, tvars_no_embedding))
trian_op_no_embedding = optimizer.apply_gradients(grads_and_vars_no_embedding, global_step=global_step)

# Keep track of gradient values and sparsity (optional)
grad_summaries = []
for g, v in grads_and_vars:
    if
g is not None:
grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
grad_summaries.append(grad_hist_summary)

grad_summaries_merged = tf.summary.merge(grad_summaries)

loss_summary = tf.summary.scalar('loss', loss)
acc_summary = tf.summary.scalar('accuracy', acc)

train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
train_summary_dir = os.path.join(out_dir, "summaries", "train")
train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

# save model
checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
checkpoint_prefix = os.path.join(checkpoint_dir, "model")
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
# saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)
saver = tf.train.Saver(tf.global_variables(), max_to_keep=2)
# saver.save(sess, checkpoint_prefix, global_step=FLAGS.num_checkpoints)

# 初始化多有的变量
sess.run(tf.global_variables_initializer())


def train_step(x_batch, y_batch):
    feed_dict = {
        textcnn._inputs_x: x_batch,
        textcnn._inputs_y: y_batch,
        textcnn._keep_dropout_prob: 0.5
    }
    _, step, summaries, cost, accuracy = sess.run([train_op, global_step, train_summary_op, loss, acc], feed_dict)
    # print tf.shape(y_batch)
    # print textcnn.predictions.get_shape()
    # time_str = str(int(time.time()))
    time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, cost, accuracy))
    train_summary_writer.add_summary(summaries, step)

    return step


def train_step_no_embedding(x_batch, y_batch):
    feed_dict = {
        textcnn._inputs_x: x_batch,
        textcnn._inputs_y: y_batch,
        textcnn._keep_dropout_prob: 0.5
    }
    _, step, summaries, cost, accuracy = sess.run([train_op_no_embedding, global_step, train_summary_op, loss, acc],
                                                  feed_dict)
    time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, cost, accuracy))
    train_summary_writer.add_summary(summaries, step)

    return step


def dev_step(x_batch, y_batch, writer=None):
    feed_dict = {
        textcnn._inputs_x: x_batch,
        textcnn._inputs_y: y_batch,
        textcnn._keep_dropout_prob: 1.0
    }
    step, summaries, cost, accuracy = sess.run([global_step, dev_summary_op, loss, acc], feed_dict)
    # time_str = str(int(time.time()))
    time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("++++++++++++++++++dev++++++++++++++{}: step {}, loss {:g}, acc {:g}".format(time_str, step, cost, accuracy))
    if writer:
        writer.add_summary(summaries, step)


for epoch in range(FLAGS.num_epochs):
    print('current epoch %s' % (epoch + 1))
    for i in range(0, train_sample_n, FLAGS.batch_size):

        x = train_x[i:i + FLAGS.batch_size]
        y = train_y[i:i + FLAGS.batch_size]
        step = train_step(x, y)
        if step % FLAGS.evaluate_every == 0:
            dev_step(dev_x, dev_y, dev_summary_writer)

        if step % FLAGS.checkpoint_every == 0:
            path = saver.save(sess, checkpoint_prefix, global_step=FLAGS.num_checkpoints)
            print("Saved model checkpoint to {}\n".format(path))
