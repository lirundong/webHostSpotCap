# -*- coding: utf-8 -*-
# Rundong Li, UESTC

import jieba
import pandas as pd

# Load Test File #
FILE_PATH = 'data/testData.csv'  # columns: url,title,CTR,time
STOP_WORDS = {}.fromkeys([line.rstrip() for line in open('data/stopWords.txt')])

testData = pd.read_csv(FILE_PATH, parse_dates=[3]).sort_values(by=[u'time']).\
    applymap(lambda x: x.strip() if (type(x) == str) else x)
testData.index = pd.Index(range(0, len(testData)))
headerSer = testData.columns.to_series()
# drop duplicated header in csv, use returnNull to prevent printing dozen 'None'
returnNull = testData.apply(lambda x: testData.drop(
    x.name, inplace=True) if (x.equals(headerSer)) else None)
del returnNull

# Tokenlize title #
testData[u'token'] = testData[u'title'].apply(jieba.lcut)
