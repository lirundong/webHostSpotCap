# -*- coding: utf-8 -*-
# Rundong Li, UESTC

import jieba
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
import sys

reload(sys)
sys.setdefaultencoding('utf8')
myType = sys.getfilesystemencoding()


# Function: delete word in STOP_WORDS #
def remove_stop(list_in, stop_words):
    result_list = []
    for current_word in list_in:
        if (current_word.encode('utf8') not in stop_words) and (len(current_word.encode('utf8').decode('utf8')) > 2):
            result_list.append(current_word.encode('utf8'))
    return result_list


# Function: my_analyzer used in TfidfVectorizer
def my_analyzer(str_in):
    split_list = jieba.cut(str_in)
    token_list = remove_stop(split_list, STOP_WORDS)
    return token_list

# Load Test File #
FILE_PATH = 'data/testData.csv'  # columns: url,title,CTR,time
testData = pd.read_csv(FILE_PATH, parse_dates=[3]).sort_values(by=[u'time']).\
    applymap(lambda x: x.strip() if (type(x) == str) else x)
testData.index = pd.Index(range(0, len(testData)))
headerSer = testData.columns.to_series()
# drop duplicated header in csv, use returnNull to prevent printing dozen 'None'
returnNull = testData.apply(lambda x: testData.drop(
    x.name, inplace=True) if (x.equals(headerSer)) else None, axis=1)
del returnNull

# Set Stop Words #
STOP_WORDS = set([line.rstrip() for line in open('data/stopWords.txt')])
PERSONAL_STOP_WORDS = ['丁香园']  # add personal stop words here
STOP_WORDS.update(PERSONAL_STOP_WORDS)

# Tokenizing title #
PERSONAL_DICT_WORDS = ['丁香园']  # add personal dict words here
for word in PERSONAL_DICT_WORDS:
    jieba.add_word(word)
# testData[u'token'] = testData[u'title'].apply(jieba.cut).\
#    apply(remove_stop, args=(STOP_WORDS,))
# if title is full of stop words, drop it
# returnNull = testData.apply(lambda x: testData.drop(
#     x.name, inplace=True) if (len(x[u'token']) == 0) else None, axis=1)
# del returnNull

# Feature extraction from title #
# parameters
TIME_DELTA = pd.Timedelta('14 days')
CLUSTER_N = 3
TAG_N = 10
VERBOSE = True  # enable verbose mode while clustering?
RESULT_PATH = 'data/result.txt'
# Vectorizer and Cluster
vectorizer = TfidfVectorizer(min_df=1, analyzer=my_analyzer)
startDate = min(testData[u'time'])
stopDate = max(testData[u'time'])
resultFile = open(RESULT_PATH, 'w')
currentDate = startDate
eventIdx = 1
# loop per TIME_DELTA days:
while currentDate < (stopDate - TIME_DELTA):
    currentStop = currentDate + TIME_DELTA
    idx = testData[u'time'].apply(lambda x: x >= currentDate and x <= currentStop)
    # currentDate += pd.Timedelta('1 day')
    currentDate = currentStop
    # idx = ((testData[u'time'] >= currentDate) and (testData[u'time'] <= currentStop))
    if (idx.value_counts()[True] == 0):
        continue
    currentData = testData[idx][u'title'].tolist()
    try:
        tfidfMatrix = vectorizer.fit_transform(currentData)
    except ValueError:
        continue
    # cluster data
    km = MiniBatchKMeans(n_clusters=CLUSTER_N, init='k-means++', n_init=1,
                         init_size=1000, batch_size=1000, verbose=VERBOSE)
    km.fit(tfidfMatrix)
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]
    # extract tags
    terms = vectorizer.get_feature_names()
    currentTags = {}
    for i in range(CLUSTER_N):
        idx = order_centroids[i, :TAG_N]
        clusterTags = [terms[tag] for tag in idx]
        currentTags.update({i: clusterTags})
    # print result
    resultFile.write('时段 %d\n起始日期: %s\t终止日期: %s\n' % (eventIdx, currentDate - TIME_DELTA, currentStop))
    for i in range(CLUSTER_N):
        for j in range(min(TAG_N, len(currentTags[i]))):
            if j == 0:
                resultFile.write('事件 %d, 关键词: %s' % (i, currentTags[i][j],))
            else:
                resultFile.write(', %s' % (currentTags[i][j],))
        resultFile.write(';\n')
    resultFile.write('====================\n')
    eventIdx += 1
resultFile.close()
