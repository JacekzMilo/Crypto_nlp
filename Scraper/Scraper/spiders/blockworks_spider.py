import scrapy
import logging
from ..items import ZrozumiecbitcSpiderItem, BitcoinSpiderItem
import os.path
from pathlib import Path
import sys



class BlockWorksSpider(scrapy.Spider):
    # Name of the spider as mentioned in the "genspider" command
    name = 'block_works_spider'
    # Domains allowed for scraping, as mentioned in the "genspider" command
    allowed_domains = ['www.blockworks.co', 'blockworks.co']
    # URL(s) to scrape as mentioned in the "genspider" command
    # The scrapy spider, starts making  requests, to URLs mentioned here
    start_urls = ['https://blockworks.co/category/markets']
    custom_settings = {"FEEDS": {"Scraper/Scraper/spiders/Scraped_data.json": {"format": "jsonlines"}}, 'FEED_EXPORTERS': {
            'jsonlines': 'scrapy.exporters.JsonItemExporter'}, 'FEED_EXPORT_ENCODING': 'utf-8',
                       'USER_AGENT': "Mozilla/5.0 (iPad; CPU OS 12_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148"}

    def parse(self, response):
        block_works_spider = response.xpath('//*[@id="ajax-load-more"]/div[1]/div[1]/div')
        # for bitc in bitcoin_spider:
        title = block_works_spider.xpath(".//a//text()").get()
        link = block_works_spider.xpath(".//a//@href").get()
        # yield {"link": link, "title": title}
        # yield {"meta": response}
        yield response.follow(url=link, callback=self.parse_block_works_spider, meta={'block_works_spider_title': title})


    def parse_block_works_spider(self, response):
        item = BitcoinSpiderItem()
        item['article_name'] = response.request.meta['block_works_spider_title']
        articles = response.xpath('(/html/body/div[3]/div[2]/div[2]/div[2]')
        # for article in articles:
        item['article_text'] = articles.xpath(".//p//text()").getall()
        item['article_text']=[i.replace("\t", "").replace("\n", "") for i in item['article_text']]
        item['article_text']=[i.replace("\u00A0", " ") for i in item['article_text']]


        yield item
        logging.info(response.url)