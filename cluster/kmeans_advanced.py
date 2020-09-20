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
from cluster.kmeans_config import Config


def number_normalizer(tokens):
    """ Map all numeric tokens to a placeholder.

    For many applications, tokens that begin with a number are not directly
    useful, but the fact that such a token exists can be relevant.  By applying
    this form of dimensionality reduction, some methods may perform better.
    """
    return ("#NUMBER" if token[0].isdigit() else token for token in tokens)


class NumberNormalizingVectorizer(TfidfVectorizer):
    def build_tokenizer(self):
        tokenize = super(NumberNormalizingVectorizer, self).build_tokenizer()
        return lambda doc: list(number_normalizer(tokenize(doc)))


def get_stopwords():
    return [line.strip() for line in open('../datasets/ocean_news/cn_stopwords.txt', encoding='utf-8').readlines()]


def get_divided_data():
    data = pd.read_csv('../datasets/ocean_news/ocean_news.csv')
    data = data.drop_duplicates('CleanedContent')

    jieba.enable_parallel()
    data['DividedContent'] = data['CleanedContent'].apply(lambda i: jieba.lcut(i))

    docs = data['DividedContent']
    data['DividedContent'] = [' '.join(i) for i in docs]
    return data


def extract_traits(data, stop_words):
    print("使用稀疏向量（Sparse Vectorizer）从训练集中抽取特征")
    t0 = time.time()

    opt = Config()

    if opt.use_hashing:
        if opt.use_idf:
            hasher = HashingVectorizer(n_features=opt.n_features,
                                       stop_words=stop_words, alternate_sign=False,
                                       norm=None, binary=False)
            vectorizer = make_pipeline(hasher, TfidfTransformer())
        else:
            vectorizer = HashingVectorizer(n_features=opt.n_features,
                                           stop_words=stop_words,
                                           alternate_sign=False, norm='l2',
                                           binary=False)
    else:
        vectorizer = NumberNormalizingVectorizer(max_df=0.5, max_features=opt.n_features,
                                                 min_df=2, stop_words=stop_words, ngram_range=(1, 2),
                                                 use_idf=opt.use_idf)

    X = vectorizer.fit_transform(data['DividedContent'])
    print("完成所耗费时间： %fs" % (time.time() - t0))
    print("样本数量: %d, 特征数量: %d" % X.shape)

    if opt.n_components:
        print("用LSA进行维度规约（降维）")
        t0 = time.time()

        # Vectorizer的结果被归一化，这使得KMeans表现为球形k均值（Spherical K-means）以获得更好的结果。
        # 由于LSA / SVD结果并未标准化，我们必须重做标准化。

        svd = TruncatedSVD(opt.n_components)
        normalizer = Normalizer(copy=False)
        lsa = make_pipeline(svd, normalizer)

        X = lsa.fit_transform(X)

        print("完成所耗费时间： %fs" % (time.time() - t0))

        explained_variance = svd.explained_variance_ratio_.sum()
        print("SVD解释方差的step: {}%".format(
            int(explained_variance * 100)))

        print('特征抽取完成！')

    print('特征抽取完成！')
    return X, vectorizer, svd


def elbow_method():
    opt = Config()
    data = get_divided_data()
    stop_words = get_stopwords()

    X, _, _ = extract_traits(data, stop_words)

    wcss = []
    for i in range(1, opt.n_clusters):
        if opt.minibatch:
            km = MiniBatchKMeans(n_clusters=i, init='k-means++', n_init=2,
                                 init_size=1000, batch_size=1500, verbose=opt.verbose)
        else:
            km = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=2,
                        verbose=opt.verbose)
        km.fit(X)
        wcss.append(km.inertia_)

    plt.plot(range(1, opt.n_clusters), wcss)
    plt.title('肘 部 方 法')
    plt.xlabel('聚类的数量')
    plt.ylabel('wcss')
    plt.show()


def news_cluster_advanced():
    opt = Config()

    data = get_divided_data()
    stop_words = get_stopwords()

    X, vectorizer, svd = extract_traits(data, stop_words)

    labels = data['DividedContent']
    true_k = 31  # 聚类数量

    if opt.minibatch:
        km = MiniBatchKMeans(n_clusters=true_k, init='k-means++', n_init=2,
                             init_size=1000, batch_size=1500, verbose=opt.verbose)
    else:
        km = KMeans(n_clusters=true_k, init='k-means++', max_iter=300, n_init=5, n_jobs=-1,
                    verbose=opt.verbose)

    print("对稀疏数据（Sparse Data） 采用 %s" % km)
    t0 = time.time()
    km.fit(X)
    print("完成所耗费时间：%0.3fs" % (time.time() - t0))
    print()

    print("Homogeneity值: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
    print("Completeness值: %0.3f" % metrics.completeness_score(labels, km.labels_))
    print("V-measure值: %0.3f" % metrics.v_measure_score(labels, km.labels_))
    print("Adjusted Rand-Index值: %.3f"
          % metrics.adjusted_rand_score(labels, km.labels_))
    print("Silhouette Coefficient值: %0.3f"
          % metrics.silhouette_score(X, km.labels_, sample_size=1000))

    print()

    # 用训练好的聚类模型反推文档的所属的主题类别
    label_prediction = km.predict(X)
    label_prediction = list(label_prediction)

    if not opt.use_hashing:
        print("每个聚类的TOP关键词:")

        if opt.n_components:
            original_space_centroids = svd.inverse_transform(km.cluster_centers_)
            order_centroids = original_space_centroids.argsort()[:, ::-1]
        else:
            order_centroids = km.cluster_centers_.argsort()[:, ::-1]

        terms = vectorizer.get_feature_names()
        for i in range(true_k):
            print("簇群 %d   " % (i + 1), end='')
            print("该簇群所含文档占比为", '%.4f%%' % (int(label_prediction.count(i)) / int(len(data['DividedContent']))))
            print("簇群关键词：")
            for ind in order_centroids[i, :80]:
                print(' %s,' % terms[ind], end='')
            print('\n------------------------------------------------------------------------------------------------')


if __name__ == '__main__':
    news_cluster_advanced()