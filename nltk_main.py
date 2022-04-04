from nltk.corpus import stopwords
import urllib.request
import nltk
import json
import csv
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from Crypto_nlp.Load_to_GCP import load
import googletrans
from googletrans import Translator
# html = response.read()
from Crypto_nlp.nltk_test import fdist, result

pd.options.display.max_colwidth = 5000000
# file = (
#     "C:/Users/Jacklord/PycharmProjects/Crypto_nlp/Crypto_nlp/Scraper/Scraper/spiders/Scraped_data_eng.json")
# records = map(json.loads, open(file, encoding="utf8"))
# df = pd.DataFrame.from_records(records)
#
# text = df.loc[1:, ["text"]]
# text_str = text.to_string()
# text_str= text_str.lower()
#
#
#
# def unique_list(l):
#     ulist = []
#     [ulist.append(x) for x in l if x not in ulist]
#     return ulist
#
# text_str=' '.join(unique_list(text_str.split()))
#
# # print(text_str)
#
# # tu jest tłumaczenie na angielski
# translator = Translator()
# text_str_translated = translator.translate(text_str, src='auto', dest='en')
# text_str= text_str_translated.text
# #########
#
# tokenizer = RegexpTokenizer(r"\w+")
# text_str= tokenizer.tokenize(text_str)
#
# # print(text_str)
#
# text_str = [x for x in text_str if not (x.isdigit()
#                                          or x[0] == '-' and x[1:].isdigit())]
# # print(no_integers)
# def listToString(text_str):
#     # initialize an empty string
#     str1 = " "
#     return (str1.join(text_str))
#
#
# text_str = listToString(text_str)
#
# print(text_str)
#
#
#
# text_strip = text_str.strip()
#
# tokens = [t for t in text_strip.split()]
#
# sr = stopwords.words('english')
# clean_tokens = tokens[:]
# for token in tokens:
#     if token in stopwords.words('english'):
#         clean_tokens.remove(token)
# freq = nltk.FreqDist(clean_tokens)
#
# # dic = {}
# keys = []
# values = []
# for key, val in freq.items():
#     keys.append('{}'.format(key))
#     values.append('{}'.format(val))
#
# # print(keys)
# # dic = dict(zip(keys, values))
# df = pd.DataFrame(list(zip(keys, values)),
#                columns =['text', 'word_count'])
# print(df)
keys = fdist.keys()
values = fdist.values()
df_fdist = pd.DataFrame(list(zip(keys, values)), columns=['text', 'word_count'])

df_result = pd.DataFrame(result, columns=['X', 'Y'])
# print(df_fdist)
# print(keys)
# print(df_result)


df_fdist.to_csv(r'C:/Users/Jacklord/PycharmProjects/Crypto_nlp/Crypto_nlp/Scraper/Scraper/spiders/nltk_frequency.csv', index = False, header=True)
df_result.to_csv(r'C:/Users/Jacklord/PycharmProjects/Crypto_nlp/Crypto_nlp/Scraper/Scraper/spiders/nltk_semantics.csv', index = False, header=True)

# filename = 'C:/Users/Jacklord/PycharmProjects/Crypto_nlp/Crypto_nlp/Scraper/Scraper/spiders/nltk_refactored_text.csv'
filename1 = 'C:/Users/Jacklord/PycharmProjects/Crypto_nlp/Crypto_nlp/Scraper/Scraper/spiders/nltk_frequency.csv'
filename2 = 'C:/Users/Jacklord/PycharmProjects/Crypto_nlp/Crypto_nlp/Scraper/Scraper/spiders/nltk_semantics.csv'
load(filename1, 'article_word_count')
load(filename2, 'semantics_coordinates')