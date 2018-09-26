# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 18:17:11 2018
@author: Moc
"""

"""
基于gensim模块的中文句子相似度计算、

过程：
1.文本预处理：中文分词，去除停用词
2.计算词频
3.创建字典（单词与编号之间的映射）
4.将待比较的文档转换为向量（词袋表示方法）
5.建立语料库
6.初始化模型
7.创建索引
8.相似度计算并返回相似度最大的文本

算法：
余弦相似度
"""

import pandas as pd
import jieba
from Utils import _self_gensim_similarities
from Config import sim_min,sim_max
import time

def _self_test(documents,new_doc,stopwords,dictionary,tfidf,index):
    # 待比较的文档进行预处理
    new_doc = new_doc.replace('\u3000','').replace('\n','').replace('\u3000','')
    words = ' '.join(jieba.cut(new_doc)).split(' ')
    new_text = []
    for word in words:
        if word not in stopwords:
            new_text.append(word)
#    print(new_text)
    
    # 将待比较的文档转换为向量（词袋表示方法）
    # 使用doc2bow方法对每个不同单词的词频进行了统计，并将单词转换为其编号，然后以稀疏向量的形式返回结果
    new_vec = dictionary.doc2bow(new_text)
#    print(new_vec)

    # 相似度计算并返回相似度最大的文本
    new_vec_tfidf = tfidf[new_vec]  # 将待比较文档转换为tfidf表示方法
    
    # 计算要比较的文档与语料库中每篇文档的相似度
    sims = index[new_vec_tfidf]
#    print(sims)
    sims_list = sims.tolist()
#    print(sims_list)
    return sims_list
    
    
def _self_sim_result(dataframe):
    dataframe_ = dataframe.copy()
    self_gensim_similarities = _self_gensim_similarities()
    documents = list(dataframe['title'])
#    documents = list(dataframe['content'])
    new_doc_list = documents.copy()
    stopwords,texts = self_gensim_similarities._self_split_sentence(documents)
    texts = self_gensim_similarities._self_count_freq(texts)
    dictionary = self_gensim_similarities._self_dict(texts)
    corpus = self_gensim_similarities._self_bow(dictionary,texts)
    tfidf,corpus_tfidf = self_gensim_similarities._self_tfidf(corpus)
    index = self_gensim_similarities._self_index(corpus_tfidf)
    i = -1
    count = 0
    print('相似度筛选的阈值范围：{}-{}'.format(sim_min,sim_max),'\n')
    for new_doc in new_doc_list:
        i = i + 1
        list_sim_sentence = []
        sims_list = _self_test(documents,new_doc,stopwords,dictionary,tfidf,index)
        for result in sims_list:
            if result > sim_min and result < sim_max:
                count = count + 1
                list_sim_sentence.append("{}  {}， 相似度为：{}".format(sims_list.index(result), documents[sims_list.index(result)],result))
                if documents[sims_list.index(result)] in new_doc_list:
                    new_doc_list.remove(documents[sims_list.index(result)])
                    dataframe_ = dataframe_.drop([sims_list.index(result)])
                else:
                    continue
        if list_sim_sentence != []:
            # print('第%s个标题'%i)  
            print(i, '',new_doc.replace('\u3000','').replace('\n','').replace('\u3000',''))
            print('相似的文本有：')
            print(list_sim_sentence)
            print("")
            print("")
    print('总共剔除了 {} 个重复标题'.format(count))
    return dataframe_
    
    
if __name__ == "__main__":
    start_time = time.time()
    dataframe = pd.read_excel('./news.xlsx')
    dataframe_ = _self_sim_result(dataframe)
    end_time = time.time()
    print(end_time-start_time, 's')
    print(len(dataframe_))
#    print(dataframe_.describe())









