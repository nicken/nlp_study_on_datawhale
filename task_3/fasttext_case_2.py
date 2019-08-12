# _*_coding:utf-8 _*_
# from:https://blog.csdn.net/lxg0807/article/details/52960072
# 由于未来得及下测试数据集，所以测试一块做了一定的调整或忽略

# 导入模块
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import fasttext

import jieba
import os

basedir = "task_3/data"  # 这是我的文件地址，需跟据文件夹位置进行更改
os.chdir(basedir)
dir_list = ['affairs', 'constellation', 'economic', 'edu', 'ent', 'fashion', 'game', 'home', 'house', 'lottery',
            'science', 'sports', 'stock']

##生成fastext的训练和测试数据集

ftrain = open("news_fasttext_train.txt", "w")
ftest = open("news_fasttext_test.txt", "w")

# 第一步获取分类文本，文本直接用的清华大学的新闻分本，可在文本系列的第三篇找到下载地址。
# 输出数据格式： 样本 + 样本标签
# 说明：这一步不是必须的，可以直接从第二步开始，第二步提供了处理好的文本格式。写这一步主要是为了记忆当时是怎么处理原始文本的。
num = -1
for e in dir_list:
    num += 1
    indir = basedir + e + '/'
    files = os.listdir(indir)
    count = 0
    for fileName in files:
        count += 1
        filepath = indir + fileName
        with open(filepath, 'r') as fr:
            text = fr.read()
        text = text.decode("utf-8").encode("utf-8")
        seg_text = jieba.cut(text.replace("\t", " ").replace("\n", " "))
        outline = " ".join(seg_text)
        outline = outline.encode("utf-8") + "\t__label__" + e + "\n"
        #         print outline
        #         break

        if count < 10000:
            ftrain.write(outline)
            ftrain.flush()
            continue
        elif count < 20000:
            ftest.write(outline)
            ftest.flush()
            continue
        else:
            break

ftrain.close()
# ftest.close()


# 训练模型
classifier = fasttext.train_supervised("news_fasttext_train.txt", label_prefix="__label__")

# 保存模型
opt="/home/nicken/NLP_study/nlp_task/task_3/model/news_fasttext.model"
classifier.save_model(opt)
#load训练好的模型
#classifier = fasttext.load_model('news_fasttext.model.bin', label_prefix='__label__')


#测试模型
result = classifier.test("news_fasttext_train.txt")
# print(result.precision)
# print(result.recall)
print(result)



