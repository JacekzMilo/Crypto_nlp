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
# from Crypto_nlp.nltk_test import fdist, result
from time import sleep
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
#df_absa_scores


# print("final_df", final_df)
# final_df.to_csv(r'C:/Users/Jacklord/PycharmProjects/Crypto_nlp/Crypto_nlp/Scraper/Scraper/spiders/article_semantics_results.csv', index = False, header=True)
# sleep(1)
# filename = 'C:/Users/Jacklord/PycharmProjects/Crypto_nlp/Crypto_nlp/Scraper/Scraper/spiders/article_semantics_results.csv'
# load(filename, 'article_semantics_results')

# nltk_df2.to_csv(r'C:/Users/Jacklord/PycharmProjects/Crypto_nlp/Crypto_nlp/Scraper/Scraper/spiders/article_semantics_results_for_plot.csv', index = False, header=True)

# filename = 'C:/Users/Jacklord/PycharmProjects/Crypto_nlp/Crypto_nlp/Scraper/Scraper/spiders/article_semantics_results_for_plot.csv'

# df_absa_scores.to_csv(r'C:/Users/Jacklord/PycharmProjects/Crypto_nlp/Crypto_nlp/Scraper/Scraper/spiders/sentence_polarity_hisogram_plot.csv', index = False, header=True)
# filename = 'C:/Users/Jacklord/PycharmProjects/Crypto_nlp/Crypto_nlp/Scraper/Scraper/spiders/sentence_polarity_hisogram_plot.csv'

# load(filename, 'sentence_polarity_hisogram_plot')



####################### Pierwsza próba zrobienia czegoś z tekstem



#######################


# #######################Tokenizacja słow
# text_strip = text_str.strip()
#
# tokens = [t for t in text_strip.split()]
# print(tokens)
# #######################
#
# ####################### Erasing stopwords
# sr = stopwords.words('english')
# clean_tokens = tokens[:]
# for token in tokens:
#     if token in stopwords.words('english'):
#         clean_tokens.remove(token)
# freq = nltk.FreqDist(clean_tokens) #Tu jest liczenie słow
# #######################
#
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


#######################To jest z Crypto_nlp.nltk_test, tam jest material z kursu, word embeddings itp
# keys = fdist.keys()
# values = fdist.values()
# df_fdist = pd.DataFrame(list(zip(keys, values)), columns=['text', 'word_count'])
#
# df_result = pd.DataFrame(result, columns=['X', 'Y'])
# print(df_fdist)
# print(keys)
# print(df_result)
#######################
#
# df_fdist.to_csv(r'C:/Users/Jacklord/PycharmProjects/Crypto_nlp/Crypto_nlp/Scraper/Scraper/spiders/nltk_frequency.csv', index = False, header=True)
# df_result.to_csv(r'C:/Users/Jacklord/PycharmProjects/Crypto_nlp/Crypto_nlp/Scraper/Scraper/spiders/nltk_semantics.csv', index = False, header=True)

# filename = 'C:/Users/Jacklord/PycharmProjects/Crypto_nlp/Crypto_nlp/Scraper/Scraper/spiders/nltk_refactored_text.csv'
# filename1 = 'C:/Users/Jacklord/PycharmProjects/Crypto_nlp/Crypto_nlp/Scraper/Scraper/spiders/nltk_frequency.csv'
# filename2 = 'C:/Users/Jacklord/PycharmProjects/Crypto_nlp/Crypto_nlp/Scraper/Scraper/spiders/nltk_semantics.csv'
# load(filename1, 'article_word_count')
# load(filename2, 'semantics_coordinates')