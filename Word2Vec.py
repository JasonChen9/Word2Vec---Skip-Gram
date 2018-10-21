import collections
import math
import os
import random
import zipfile
import numpy as np
import urllib
import tensorflow as tf

url = 'http://mattmahoney.net/dc/'
def maybe_download(filename,expected_bytes):
    if not os.path.exists(filename):
        filename,_ = urllib.request.urlretrieve(url+filename,filename)
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified',filename)
    else:
        print(statinfo.st_size)
        raise Exception(
            'Failed to verify'+filename + '. Can you get to it with a browser?'
        )
    return filename

filename = maybe_download('text8.zip',31344016)
#解压下载的压缩文件，并使用tf.compat.as_str 将数据转换成单词的列表。
def read_data(filename):
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data
words = read_data(filename)
print('Data size',len(words))
#创建词汇表，统计词频，取最多的50000个作为vocabulary。再创建一个字典将单词表存在字典里以便查询
vocabulary_size = 50000
def bulid_dataset(words):
    count = [['UNK',-1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))#取词频最多的50000个
    dictionary = dict()#生成字典
    for word,_ in count:
        dictionary[word] = len(dictionary)#对字典进行编号
    data = list()#生成数据的列表
    unk_count = 0
    for word in words:#遍历字典
        if word in dictionary:#如果在字典返回单词的序列
            index = dictionary[word]
        else :#如果不在序列设为0 并统计有多少不在单词表里的单词
            index = 0
            unk_count += 1
        data.append(index)#把得到的序列添加到数据列表里
    count[0][1] = unk_count #多少词汇不在单词表里
    reverse_dictionary = dict(zip(dictionary.values(),dictionary.keys()))#返回反转的字典
    return data,count,dictionary,reverse_dictionary
data, count,dictionary,reverse_dictionary = bulid_dataset(words)
del words#删除原始的词汇表可以节省内存
#打印vocabulary中最高频的词汇即数量
print('Most common words (+UNK)',count[:5])
print('Sample data',data[:10],[reverse_dictionary[i] for i in data[:10]])

#下面来生成Word2Vec的训练样本。


data_index = 0
#来生成训练用的batch数据 batch_size为batch大小 num_skips是每个单词生成多少个样本 skip_window指最远可以联系到的单词
def generate_batch(batch_size , num_skips , skip_window):
    global data_index#定义为全局变量
    assert batch_size % num_skips == 0#使batch_size是num_skips的整数倍
    assert num_skips <= 2 * skip_window#num_skips不能超过窗口的二倍
    batch = np.ndarray(shape=(batch_size),dtype=np.int32)#初始化为数组
    labels = np.ndarray(shape=(batch_size,1),dtype=np.int32)#初始化为数组
    span = 2 * skip_window + 1#对某个单词创建相关样本时会用到的单词数量
    buffer = collections.deque(maxlen=span)#创建一个最大容量为span的双向队列
    for _ in range(span):#从data_index开始把span个单词顺序读入buffer作为初始值
        buffer.append(data[data_index])
        data_index = (data_index + 1)% len(data)
    for i in range(batch_size // num_skips):
        target = skip_window #目标单词是第skip_window个
        targets_to_avoid =[ skip_window ]#生成样本时需要避免的样本
        for j in range(num_skips):#生成每个单词的样本
            while target in targets_to_avoid:#先产生随机数，直到随机数不在避免的列表里
                 target = random.randint(0,span - 1)
            targets_to_avoid.append(target)#把单词加入到避免的列表
            batch[i * num_skips +j]=buffer[skip_window]
            labels[i * num_skips + j , 0] =buffer[target]
        buffer.append(data[data_index])#加入队列一个新单词抛弃buffer的第一个单词 同时语境向后移动一位
        data_index = (data_index + 1)%len(data)
    return batch,labels
#测试功能
batch,labels = generate_batch(batch_size=8,num_skips=2,skip_window=1)
for i in range(8):
    print(batch[i],reverse_dictionary[batch[i]],'->',labels[i,0],reverse_dictionary[labels[i,0]])

batch_size = 128
embedding_size = 128#单词的嵌入向量
skip_window = 1
num_skips = 2
valid_size = 16#抽取的验证单词数
valid_window = 100#从词频最高的100个单词里抽
valid_examples = np.random.choice(valid_window,valid_size,replace=False)
num_sampled = 64#负采样个数

graph = tf.Graph()
with graph.as_default():
    train_inputs = tf.placeholder(tf.int32,shape=[batch_size])
    train_labels = tf.placeholder(tf.int32,shape=[batch_size,1])
    valid_dataset = tf.constant(valid_examples,dtype=tf.int32)

    with tf.device('/cpu:0'):
        embeddings = tf.Variable(
            tf.random_uniform([vocabulary_size,embedding_size],-1.0,1.0))#生成50000行128列分布在-1到1的随机矩阵
    embed = tf.nn.embedding_lookup(embeddings,train_inputs)#从嵌入矩阵得到训练集的嵌入

    nce_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size,embedding_size],stddev=1.0/math.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

    loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights, biases=nce_biases, labels=train_labels, inputs=embed,num_sampled=num_sampled, num_classes=vocabulary_size))
    #使用随机梯度下降 学习速率为1.0
    optimzer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
    #计算嵌入向量的L2范数
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings),1,keep_dims=True))
    #除以l2范数 得到标准化的向量
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(
    normalized_embeddings,valid_dataset)#查询验证单词的嵌入向量
    similarity = tf.matmul(
    valid_embeddings,normalized_embeddings,transpose_b=True
)#计算相似性
    init = tf.global_variables_initializer()

#我们定义最大迭代次数为10万次
num_steps = 100001
with tf.Session(graph=graph) as session:
    init.run()
    print("Initialized")

    average_loss = 0
    for step in range(num_steps):
        #生成一个batch的数据
        batch_inputs , batch_labels = generate_batch(
            batch_size,num_skips,skip_window)
        feed_dict = {train_inputs : batch_inputs,train_labels:batch_labels}
        #使用session.run()执行一次优化器
        _,loss_val = session.run([optimzer,loss],feed_dict=feed_dict)
        average_loss += loss_val #并把这一步训练的loss累积到average loss

        if step % 2000 == 0:
            if step > 0:
                average_loss /= 2000
            print("Average loss at step", step, ":", average_loss)
            average_loss = 0
        #每10000次循环，计算一次验证单词与全部单词的相似度，并将与每个单词最相似的8个单词展示出来
        if step % 10000 == 0:
            sim = similarity.eval()
            for i in range(valid_size):
                valid_word = reverse_dictionary[valid_examples[i]]
                top_k = 8
                nearest = (-sim[i, :]).argsort()[1:top_k+1]
                log_str = "Nearest to %s:" % valid_word
                for k in range(top_k):
                    closs_word = reverse_dictionary[nearest[k]]
                    log_str = "%s %s ," % (log_str,closs_word)
                print(log_str)
    final_embeddings = normalized_embeddings.eval()
#把训练结果可视化并保存在本地
def plot_with_labels(low_dim_embs,labels,filename='tsne500.png'):
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    plt.figure(figsize=(18,18))
    for i ,label in enumerate(labels):
        x,y=low_dim_embs[i,:]
        plt.scatter(x,y)
        plt.annotate(label,xy=(x,y),xytext=(5,2),textcoords='offset points',ha='right',va='bottom')
    plt.savefig(filename)
#采用TSNE降维 将原始的128维下降到2维
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
tsne = TSNE(perplexity=30,n_components=2,init='pca',n_iter=5000)
plot_only = 500
#显示词频最高的100个单词
low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only,:])
labels = [reverse_dictionary[i] for i in range(plot_only)]
plot_with_labels(low_dim_embs,labels)





