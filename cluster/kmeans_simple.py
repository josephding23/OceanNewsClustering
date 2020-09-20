import json
import unicodedata
import jieba.posseg as pseg
import jieba
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics
from pprint import pprint
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import time


def get_ocean_stopwords():
    stopwords_path = '../datasets/ocean_news/stopwords.json'
    stopwords = []
    with open(stopwords_path, 'r') as load_f:
        stopwords_dict = json.load(load_f)
        for stopword_info in stopwords_dict:
            stopwords.append(stopword_info['Keyword'])
    return stopwords


def get_stopwords():
    return [line.strip() for line in open('../datasets/ocean_news/cn_stopwords.txt', encoding='utf-8').readlines()]


def get_news_titles():
    news_path = '../datasets/ocean_news/ocean_news.json'
    titles = []
    with open(news_path, 'r') as load_f:
        news_dict = json.load(load_f)
        for news_info in news_dict:
            titles.append(news_info['Article'])
    return titles


def get_news_contents():
    news_path = '../datasets/ocean_news/ocean_news.json'
    contents = []
    with open(news_path, 'r') as load_f:
        news_dict = json.load(load_f)
        for news_info in news_dict:
            raw_content = news_info['CleanedContent']
            content = unicodedata.normalize('NFKC', raw_content)
            contents.append(content)
    return contents


def jieba_cut(comment):
    word_list = []
    seg_list = pseg.cut(comment)
    for word in seg_list:
        if word.flag in ['ns', 'n', 'vn', 'v', 'nr', 'nt', 'nz']:
            word_list.append(word.word)
    return word_list


def news_cluster_simple():
    comment_list = get_news_titles()
    vectorizer = TfidfVectorizer(stop_words=get_stopwords(), tokenizer=jieba_cut, use_idf=True)  # 创建词向量模型
    X = vectorizer.fit_transform(comment_list)  # 将评论关键字列表转换为词向量空间模型
    # K均值聚类
    model_kmeans = KMeans(n_clusters=10)  # 创建聚类模型对象
    model_kmeans.fit(X)  # 训练模型

    cluster_labels = model_kmeans.labels_  # 聚类标签结果
    word_vectors = vectorizer.get_feature_names()  # 词向量
    word_values = X.toarray()  # 向量值
    comment_matrix = np.hstack((word_values, cluster_labels.reshape(word_values.
                                                                    shape[0], 1)))  # 将向量值和标签值合并为新的矩阵
    word_vectors.append('cluster_labels')  # 将新的聚类标签列表追加到词向量后面
    comment_pd = pd.DataFrame(comment_matrix, columns=word_vectors)  # 创建包含词向量和聚类标签的数据框
    # comment_pd.to_csv('../models/clustering_results/comment.csv')
    print(comment_pd.head(1))  # 打印输出数据框第1条数据
    # 聚类结果分析

    comment_cluster1 = comment_pd[comment_pd['cluster_labels'] == 1].drop('cluster_labels', axis=1)
    comment_cluster2 = comment_pd[comment_pd['cluster_labels'] == 2].drop('cluster_labels', axis=1)
    comment_cluster3 = comment_pd[comment_pd['cluster_labels'] == 3].drop('cluster_labels', axis=1)
    comment_cluster4 = comment_pd[comment_pd['cluster_labels'] == 4].drop('cluster_labels', axis=1)
    comment_cluster5 = comment_pd[comment_pd['cluster_labels'] == 5].drop('cluster_labels', axis=1)

    word_importance1 = np.sum(comment_cluster1, axis=0)
    word_importance2 = np.sum(comment_cluster2, axis=0)
    word_importance3 = np.sum(comment_cluster3, axis=0)
    word_importance4 = np.sum(comment_cluster4, axis=0)
    word_importance5 = np.sum(comment_cluster5, axis=0)

    print(word_importance1.sort_values(ascending=False)[:5])
    print(word_importance2.sort_values(ascending=False)[:5])
    print(word_importance3.sort_values(ascending=False)[:5])
    print(word_importance4.sort_values(ascending=False)[:5])
    print(word_importance5.sort_values(ascending=False)[:5])




if __name__ == '__main__':
    news_cluster_advanced()
