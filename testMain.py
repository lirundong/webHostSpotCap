# -*- coding: utf-8 -*-
# Rundong Li, UESTC

import pandas as pd
import jieba.analyse
import jieba
import json
import sys

reload(sys)
sys.setdefaultencoding('utf8')
myType = sys.getfilesystemencoding()


# FUNCTIONS #
def count_period(date, begin_date, const_date):
    date_delta = date - begin_date
    begin_idx = int((date_delta - (const_date - pd.Timedelta('1 days'))).days)
    begin_idx = begin_idx if (begin_idx > 0) else 0
    end_idx = int(date_delta.days)
    return range(begin_idx, end_idx)


def hot_topic_under_tag(vec_list, tags, top_k):
    topic_dict = {}
    for tag in tags:
        include_tag_idx = vec_list[u'vecLst'].apply(lambda x: tag in x)
        topic_idx = vec_list.ix[include_tag_idx].sort_values(
            by=[u'CTR'], ascending=False).iloc[0:top_k].index
        tag_dict = {}
        for Idx in topic_idx:
            tag_dict.update(
                {'title': unicode(vec_list.ix[Idx, u'title']), 'URL': vec_list.ix[Idx, u'url']})
        topic_dict.update({unicode(tag): tag_dict})
    return topic_dict


def one_hot_coding(list_in, token_list, CTR=1):
    series_in = pd.Series(list_in)
    token_count = series_in.value_counts() * CTR
    token_count.apply()
    return [token in list_in for token in token_list]


# READ DATA #
testDataPath = 'data/testData.csv'  # columns: url,title,CTR,time
testData = pd.read_csv(testDataPath, parse_dates=[3], nrows=100).sort_values(by=[u'time']).\
    applymap(lambda x: x.strip() if (type(x) == str) else x)
testData.index = pd.Index(range(0, len(testData)))
headerSer = testData.columns.to_series()
# drop duplicated header in csv, use returnNull to prevent printing dozen
# 'None'
returnNull = testData.apply(lambda x: testData.drop(
    x.name, inplace=True) if (x.equals(headerSer)) else None)
del returnNull

# READ STOP-WORDS #
stopWordsPath = 'data/stopWords.txt'
f = open(stopWordsPath, 'r')
stopList = f.readlines()
f.close()

# ADJUSTABLE PARAMETERS #
topK_eachDay = 3  # extract 3 tags per day
topK_eachTag = 3  # extract 3 hottest topics under each tag
beginDay = testData.ix[0, u'time']
hotConstDays = pd.Timedelta('14 days')  # each hot-spot const for 14 days

# express each record in a <vector generator>
testData[u'vecLst'] = testData[u'title'].apply(jieba.lcut)

# loop though 14 day per time, extract tags
testData[u'eventIdx'] = testData[u'time'].apply(
    count_period, args=(beginDay, hotConstDays))
eventDict = {}

# loop for each event
for eventIdx in range(max(testData[u'eventIdx'].apply(lambda x: max(x) if len(x) > 0 else 0))):
    eventDateIdx = testData[testData[u'eventIdx'].apply(
        lambda x: eventIdx in x)].index
    if len(eventDateIdx) == 0:
        continue
    startDate = testData.ix[eventDateIdx, u'time'].iloc[0]
    endDate = testData.ix[eventDateIdx, u'time'].iloc[-1]

    # cluster each event
    # one-hot encoding
    tokenList = []
    returnNull = testData.ix[eventDateIdx, u'vecLst'].apply(lambda x: tokenList.extend(x))
    del returnNull
    tokenList = list(set(tokenList) - set(stopList))

    # extract tags
    eventStr = ''.join(testData.ix[eventDateIdx, u'title'])
    eventTags = jieba.analyse.extract_tags(eventStr, topK=topK_eachDay)
    topicDict = hot_topic_under_tag(
        testData.ix[eventDateIdx, :], eventTags, topK_eachTag)
    eventDict.update({u'event_' + str(eventIdx): {'hot_topic': topicDict,
                                                  'start_date': unicode(startDate), 'end_date': unicode(endDate)}})

# SHOW RESULT #
jsonPath = 'result/result.json'
resultJSON = json.dumps(eventDict)

jsonFile = open(jsonPath, 'w')
jsonFile.write(resultJSON)
jsonFile.close()
