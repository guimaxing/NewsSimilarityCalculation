# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 18:17:11 2018
@author: Moc
"""

from gensim import corpora, models, similarities
from collections import defaultdict
import jieba

class _self_gensim_similarities:

    def _self_split_sentence(self, documents):
        # 文本预处理：中文分词，去除停用词, documents type:list
        print('文本预处理：中文分词，去除停用词')
        # 获取停用词
        stopwords = set()
        file = open("stopwords.txt", 'r', encoding='UTF-8')
        for line in file:
            stopwords.add(line.strip())
        file.close()
        
        # 将分词、去停用词后的文本数据存储在list类型的texts中
        texts = []
        for line in documents:
            line = line.replace('\u3000','').replace('\n','').replace('\u3000','')
            words = ' '.join(jieba.cut(line)).split(' ')    # 利用jieba工具进行中文分词
            text = []
            # 过滤停用词，只保留不属于停用词的词语
            for word in words:
                if word not in stopwords:
                    text.append(word)
            texts.append(text)
#        for line in texts:
#            print(line)
        return stopwords,texts
    
    def _self_count_freq(self, texts):
        # 计算词频
        print('计算词频')
        frequency = defaultdict(int)  # 构建一个字典对象
        # 遍历分词后的结果集，计算每个词出现的频率
        for text in texts:
            for word in text:
                frequency[word] += 1
        # 选择频率大于1的词(根据实际需求确定)
        texts = [[word for word in text if frequency[word] > 1] for text in texts]
#        for line in texts:
#            print(line)
        return texts
            
          
    def _self_dict(self, texts):
        # 创建字典（单词与编号之间的映射）
        print('创建字典（单词与编号之间的映射）')
        dictionary = corpora.Dictionary(texts)
#        print(dictionary)
        # 打印字典，key为单词，value为单词的编号
#        print(dictionary.token2id)
        return dictionary
        
        
    def _self_bow(self,dictionary,texts):
        # 建立语料库
        print('建立语料库')
        # 将每一篇文档转换为向量
        corpus = [dictionary.doc2bow(text) for text in texts]
#        print(corpus)
        return corpus
        
    def _self_tfidf(self, corpus):
        # 初始化模型
        print('初始化模型 tfidf')
        # 初始化一个tfidf模型,可以用它来转换向量（词袋整数计数），表示方法为新的表示方法（Tfidf 实数权重）
        tfidf = models.TfidfModel(corpus)
        # 将整个语料库转为tfidf表示方法
        corpus_tfidf = tfidf[corpus]
#        for doc in corpus_tfidf:
#            print(doc)
        return tfidf,corpus_tfidf
            
            
    def _self_index(self, corpus_tfidf):
        # 创建索引
        print('创建索引')
        # 使用上一步得到的带有tfidf值的语料库建立索引
        index = similarities.MatrixSimilarity(corpus_tfidf)
        return index
    
    
    