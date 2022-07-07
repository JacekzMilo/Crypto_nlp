from Crypto_nlp.Scraper.Scraper.custom_text_edit import customtextfunc, article_translation
from Crypto_nlp.aspect_based_semantic_analysis import nlp_article_semantic
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings
from Crypto_nlp.Scraper.Scraper.spiders import zrozumiecbitc, bitcoin_spider, insights_glassnode_spider, coin_desk_spider, \
    bitcoin_magazine_spider, the_block_spider, the_defiant_spider, blockworks_spider
import os


# ###################### Below code runs scrapping module
# settings_file_path = 'Scraper.settings' # The path seen from root, ie. from main.py
# os.environ.setdefault('SCRAPY_SETTINGS_MODULE', settings_file_path)
# crawlers = [zrozumiecbitc.ZrozumiecbitcSpider, bitcoin_spider.BitcoinSpider,
#             insights_glassnode_spider.InsightsGlassnodeSpider,
#             coin_desk_spider.CoinDeskSpider, bitcoin_magazine_spider.BitcoinMagazineSpider
#             ]
#
# # TODO: the_block_spider.TheBlockSpider i the_defiant_spider.TheDefiantSpider i blockworks_spider.BlockWorksSpider
#
# # TODO: coindeskspider has hardcoded div range and may not cover all article
#
# def start_sequentially(process: CrawlerProcess, crawlers: list):
#     print('start crawler {}'.format(crawlers[0].__name__))
#     deferred = process.crawl(crawlers[0])
#     if len(crawlers) > 1:
#         deferred.addCallback(lambda _: start_sequentially(process, crawlers[1:]))#
#
# process = CrawlerProcess(settings=get_project_settings())
# start_sequentially(process, crawlers)
# # settings=get_project_settings()
# # process = CrawlerProcess(settings)
# # process.crawl(zrozumiecbitc.ZrozumiecbitcSpider)
# process.start()
# # print(output)
# ######################


###################### Tu by sie przydało usprawnić. Gdy raz przejdzie przez customtextfunc i ponownie chce przejsc to wyskakuje blad.
# Below code runs customtextfunc that puts scrapped text into article table in BQ

file ='C:/Users/Jacklord/PycharmProjects/Crypto_nlp/Crypto_nlp/Scraper/Scraper/spiders/Scraped_data.json'
customtextfunc(file)
#######################


####################### Below code runs article_translation function that translate article text into English and puts
# the data into article_translated table in BQ

filename = 'C:/Users/Jacklord/PycharmProjects/Crypto_nlp/Crypto_nlp/Scraper/Scraper/spiders/Scraped_data.csv'
article_translation(filename)
#######################


#######################the code below creates the sentence_polarity_hisogram_plot.csv and puts the sentence_polarity_distribution_plot into the tables on the BQ
# Then creates the file article_semantics_results_for_plot.csv which he puts in the results_for_plot table
# Then creates feature_polarity_calculations_df.csv and adds sentence_polarity_hisogram_plot to the table

nlp_article_semantic("C:/Users/Jacklord/PycharmProjects/Crypto_nlp/Crypto_nlp/Scraper/Scraper/spiders/article_translated.csv")
#######################



