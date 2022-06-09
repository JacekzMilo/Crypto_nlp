import scrapy
import logging
from ..items import ZrozumiecbitcSpiderItem, BitcoinSpiderItem
import os.path
from pathlib import Path
import sys



class BitcoinSpider(scrapy.Spider):
    # Name of the spider as mentioned in the "genspider" command
    name = 'bitcoin_spider'
    # Domains allowed for scraping, as mentioned in the "genspider" command
    allowed_domains = ['www.markets.businessinsider.com', 'markets.businessinsider.com']
    # URL(s) to scrape as mentioned in the "genspider" command
    # The scrapy spider, starts making  requests, to URLs mentioned here
    start_urls = ['https://markets.businessinsider.com/currencies/btc-usd']
    custom_settings = {"FEEDS": {"Scraper/Scraper/spiders/Scraped_data.json": {"format": "jsonlines"}}, 'FEED_EXPORTERS': {
            'jsonlines': 'scrapy.exporters.JsonItemExporter'}, 'FEED_EXPORT_ENCODING': 'utf-8'}

    def parse(self, response):
        bitcoin_spider = response.xpath('//*[@id="instrument-detail-news"]/section/div[1]/h3/a')
        for bitc in bitcoin_spider:
            title = bitcoin_spider.xpath(".//text()").get()
            link = bitcoin_spider.xpath(".//@href").get()
        # yield {"link": link, "title": title}
        # yield {"meta": response}
        yield response.follow(url=link, callback=self.parse_bitcoin_spider, meta={'bitcoin_spider_title': title})


    def parse_bitcoin_spider(self, response):
        item = BitcoinSpiderItem()
        item['article_name'] = response.request.meta['bitcoin_spider_title']
        articles = response.xpath('(//div[@id="piano-inline-content-wrapper"])')
        # for article in articles:
        item['article_text'] = articles.xpath(".//p//text()").getall()
        item['article_text']=[i.replace("\t", "").replace("\n", "") for i in item['article_text']]
        item['article_text']=[i.replace("\u00A0", " ") for i in item['article_text']]


        yield item
        logging.info(response.url)



# path = Path("C:/Users/Jacklord/PycharmProjects/Crypto_nlp/Crypto_nlp/Scraper/Scraper/spiders/Scraped_data_no_quotes1.json")
#
# if os.path.exists(path) is True:
#     print("file is already changed, updated data in BQ")
#     from ..create_table_ndjson_bq import dataload
#     dataload()

# else:
# if os.path.isfile('C:/Users/Jacklord/PycharmProjects/Crypto_nlp/Crypto_nlp/Scraper/Scraper/spiders/Scraped_data.json'):
#     from ..custom_text_edit import customtextfunc
#     from ..create_table_ndjson_bq import dataload
#     # from ..create_table_ndjson_bq import dataload
#     file = open('C:/Users/Jacklord/PycharmProjects/Crypto_nlp/Crypto_nlp/Scraper/Scraper/spiders/Scraped_data.json',
#                 "r")
#     customtextfunc(file)
#     dataload()