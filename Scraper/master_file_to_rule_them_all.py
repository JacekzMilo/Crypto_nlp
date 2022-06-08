from Crypto_nlp.Scraper.Scraper.custom_text_edit import customtextfunc, article_translation
from Crypto_nlp.Load_to_GCP import load
from Crypto_nlp.aspect_based_opinion_mining_mlp9_github import nlp_article_semantic
from scrapy.signalmanager import dispatcher
from scrapy import signals
from scrapy.crawler import CrawlerProcess
# from Crypto_nlp.Scraper.Scraper import settings
# from scrapy.settings import Settings
from scrapy.utils.project import get_project_settings
from Crypto_nlp.Scraper.Scraper.spiders import zrozumiecbitc, bitcoin_spider
import os
import pandas as pd
from time import sleep
from twisted.internet import reactor, defer

# ######################To odpala scrapera
# # output = []
# # def get_output(item):
# #     output.append(item)
# #     return output
# # dispatcher.connect(get_output, signal=signals.item_scraped)
#
# settings_file_path = 'Scraper.settings' # The path seen from root, ie. from main.py
# os.environ.setdefault('SCRAPY_SETTINGS_MODULE', settings_file_path)
# crawlers = [zrozumiecbitc.ZrozumiecbitcSpider, bitcoin_spider.BitcoinSpider]
#
#
# def start_sequentially(process: CrawlerProcess, crawlers: list):
#     print('start crawler {}'.format(crawlers[0].__name__))
#     deferred = process.crawl(crawlers[0])
#     if len(crawlers) > 1:
#         deferred.addCallback(lambda _: start_sequentially(process, crawlers[1:]))
#
#
# process = CrawlerProcess(settings=get_project_settings())
# start_sequentially(process, crawlers)
# # settings=get_project_settings()
# # process = CrawlerProcess(settings)
# # process.crawl(zrozumiecbitc.ZrozumiecbitcSpider)
# process.start()
# # print(output)
# #######################

# # sleep(0.5)
# ####################### Tu by sie przydało usprawnić. Gdy raz przejdzie przez customtextfunc i ponownie chce przejsc to wyskakuje blad.
# #wrzuca do tabeli article oryginalny text ze strony do tabeli article
#
file ='C:/Users/Jacklord/PycharmProjects/Crypto_nlp/Crypto_nlp/Scraper/Scraper/spiders/Scraped_data.json'
customtextfunc(file)

# #######################
# #######################Tlumaczy na angielski, tworzy plik article_translated.csv i wrzuca w oddzielna tabele w BQ: article_translated
#
filename = 'C:/Users/Jacklord/PycharmProjects/Crypto_nlp/Crypto_nlp/Scraper/Scraper/spiders/Scraped_data.csv'
article_translation(filename)
#
# #######################

# #######################to tworzy sentence_polarity_hisogram_plot.csv i wrzuca w tabele sentence_polarity_distribution_plot, DS na podstawie tej
# # tabeli tworzy wykres dystrybucji polarity (slupkowy+liniowy)
# # Następnie tworzy plik article_semantics_results_for_plot.csv który wrzuca w tabele results_for_plot z której powstaje wykres słupkowy pos/neg
# #Następnie tworzy plik feature_polarity_calculations_df.csv i wrzuca do tabeli sentence_polarity_hisogram_plot i z tego tworzy wykres boxplot.
#
#
nlp_article_semantic("C:/Users/Jacklord/PycharmProjects/Crypto_nlp/Crypto_nlp/Scraper/Scraper/spiders/article_translated.csv")
# nlp_article_semantic("C:/Users/Jacklord/PycharmProjects/Crypto_nlp/Crypto_nlp/Scraper/Scraper/spiders/Scraped_data_eng_ja_edytowalem_recznie.json")
# #######################


#lacznie powstaje 5 tabel w BQ


