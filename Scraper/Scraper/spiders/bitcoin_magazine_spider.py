import scrapy
import logging
from ..items import ZrozumiecbitcSpiderItem, BitcoinSpiderItem
import os.path
from pathlib import Path
import sys



class BitcoinMagazineSpider(scrapy.Spider):
    # Name of the spider as mentioned in the "genspider" command
    name = 'bitcoin_spider'
    # Domains allowed for scraping, as mentioned in the "genspider" command
    allowed_domains = ['www.bitcoinmagazine.com', 'bitcoinmagazine.com']
    # URL(s) to scrape as mentioned in the "genspider" command
    # The scrapy spider, starts making  requests, to URLs mentioned here
    start_urls = ['https://bitcoinmagazine.com/business']
    custom_settings = {"FEEDS": {"Scraper/Scraper/spiders/Scraped_data.json": {"format": "jsonlines"}}, 'FEED_EXPORTERS': {
            'jsonlines': 'scrapy.exporters.JsonItemExporter'}, 'FEED_EXPORT_ENCODING': 'utf-8'}

    def parse(self, response):
        bitcoin_magazine_spider = response.xpath('//*[@id="main-content"]/section[3]/phoenix-hub/section[1]/phoenix-non-personalized-recommendations-tracking/div[1]/phoenix-super-link/phoenix-card/div[2]/phoenix-ellipsis')
        for bitc in bitcoin_magazine_spider:
            title = bitcoin_magazine_spider.xpath(".//a/h2//text()").get()
            link = bitcoin_magazine_spider.xpath(".//a/@href").get()
        # yield {"link": link, "title": title}
        # yield {"meta": response}
        yield response.follow(url=link, callback=self.parse_bitcoin_magazine_spider, meta={'bitcoin_magazine_spider_title': title})


    def parse_bitcoin_magazine_spider(self, response):
        item = BitcoinSpiderItem()
        item['article_name'] = response.request.meta['bitcoin_magazine_spider_title']
        articles = response.xpath('(//*[@id="main-content"]/section[1]/article/div/div/section/div[1]/div[3])')
        # for article in articles:
        item['article_text'] = articles.xpath(".//ul/li//text()").getall()
        item['article_text'] = item['article_text'] + articles.xpath(".//p//text()").getall()
        item['article_text']=[i.replace("\t", "").replace("\n", "") for i in item['article_text']]
        item['article_text']=[i.replace("\u00A0", " ") for i in item['article_text']]


        yield item
        logging.info(response.url)