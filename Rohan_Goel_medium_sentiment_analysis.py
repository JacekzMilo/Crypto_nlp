import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
import stanza
import json
from nltk.tokenize import RegexpTokenizer


# pd.options.display.max_colwidth = 5000000
# file = (
#     "C:/Users/Jacklord/PycharmProjects/Crypto_nlp/Crypto_nlp/Scraper/Scraper/spiders/Scraped_data_eng.json")
# records = map(json.loads, open(file, encoding="utf8"))
# df = pd.DataFrame.from_records(records)
#
# text = df.loc[1:, ["text"]]
# txt= text.to_string()
# txt= txt.lower()
#
# def unique_list(l):
#     ulist = []
#     [ulist.append(x) for x in l if x not in ulist]
#     return ulist
#
# txt=' '.join(unique_list(txt.split()))
#
# tokenizer = RegexpTokenizer(r"\w+")
# txt= tokenizer.tokenize(txt)
#
# # print(text_str)
#
# txt = [x for x in txt if not (x.isdigit()
#                                          or x[0] == '-' and x[1:].isdigit())]
# # print(no_integers)
# def listToString(txt):
#     # initialize an empty string
#     str1 = " "
#     return (str1.join(txt))
#
#
# txt = listToString(txt)

# txt = "The Sound Quality is great but the battery life is very bad."


# txt = txt.lower()
# print(txt)


txt = "Which DeFi project to choose? Which coin to enter now?"

sentList = nltk.sent_tokenize(txt)

fcluster = []
totalfeatureList = []
finalcluster = []
dic = {}

for line in sentList:
    txt_list = nltk.word_tokenize(line)
    taggedList = nltk.pos_tag(txt_list)
print(taggedList)

newwordList = []
flag = 0
for i in range(0,len(taggedList)-1):
    if(taggedList[i][1]=="NN" and taggedList[i+1][1]=="NN"):
        newwordList.append(taggedList[i][0]+taggedList[i+1][0])
        flag=1
    else:
        if(flag==1):
            flag=0
            continue
        newwordList.append(taggedList[i][0])
        if(i==len(taggedList)-2):
            newwordList.append(taggedList[i+1][0])
finaltxt = ' '.join(word for word in newwordList)
print(finaltxt)


stop_words = set(stopwords.words('english'))
new_txt_list = nltk.word_tokenize(finaltxt)
wordsList = [w for w in new_txt_list if not w in stop_words]
taggedList = nltk.pos_tag(wordsList)

# stanza.download('en')
nlp = stanza.Pipeline('en')
doc = nlp(finaltxt)
dep_node = []
for dep_edge in doc.sentences[0].dependencies:
    dep_node.append([dep_edge[2].text, dep_edge[0].id, dep_edge[1]])
for i in range(0, len(dep_node)):
    if int(dep_node[i][1]) != 0:
        dep_node[i][1] = newwordList[(int(dep_node[i][1]) - 1)]
# print(dep_node)

featureList = []
categories = []
for i in taggedList:
    if(i[1]=='JJ' or i[1]=='NN' or i[1]=='JJR' or i[1]=='NNS' or i[1]=='RB'):
        featureList.append(list(i))
        totalfeatureList.append(list(i)) # This list will store all the features for every sentence
        categories.append(i[0])
print(featureList)
print(categories)

fcluster = []
for i in featureList:
    filist = []
    for j in dep_node:
        if((j[0]==i[0] or j[1]==i[0]) and (j[2] in ["nsubj", "acl:relcl", "obj", "dobj", "agent", "advmod", "amod", "neg", "prep_of", "acomp", "xcomp", "compound"])):
            if(j[0]==i[0]):
                filist.append(j[1])
            else:
                filist.append(j[0])
    fcluster.append([i[0], filist])
print(fcluster)

finalcluster = []
dic = {}
for i in featureList:
    dic[i[0]] = i[1]
for i in fcluster:
    if(dic[i[0]]=="NN"):
        finalcluster.append(i)
print(finalcluster)