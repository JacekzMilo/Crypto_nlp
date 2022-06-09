import scrapy
import logging
from ..items import ZrozumiecbitcSpiderItem, BitcoinSpiderItem
import os.path
from pathlib import Path
import sys



class TheDefiantSpider(scrapy.Spider):
    # Name of the spider as mentioned in the "genspider" command
    name = 'the_defiant_spider'
    # Domains allowed for scraping, as mentioned in the "genspider" command
    allowed_domains = ['www.thedefiant.io', 'thedefiant.io']
    # URL(s) to scrape as mentioned in the "genspider" command
    # The scrapy spider, starts making  requests, to URLs mentioned here
    start_urls = ['https://thedefiant.io/layer-1-solana/', 'https://thedefiant.io/layer-1-ethereum/']
    custom_settings = {"FEEDS": {"Scraper/Scraper/spiders/Scraped_data.json": {"format": "jsonlines"}}, 'FEED_EXPORTERS': {
            'jsonlines': 'scrapy.exporters.JsonItemExporter'}, 'FEED_EXPORT_ENCODING': 'utf-8',
                       'USER_AGENT': "Mozilla/5.0 (iPad; CPU OS 12_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148"}

    for i in range(len(start_urls)):
        if i == 0:
            def parse(self, response):
                the_defiant_spider = response.xpath('//*[@id="uc_post_list_elementor10262"]/div[1]/div[2]/div/div[1]')
                # for bitc in bloomberg_spider:
                title = the_defiant_spider.xpath(".//a/text()").get()
                link = the_defiant_spider.xpath(".//a/@href").get()
                # yield {"link": link, "title": title}
                # yield {"meta": response}
                yield response.follow(url=link, callback=self.parse_the_defiant_spider, meta={'the_defiant_spider_title': title})


            def parse_the_defiant_spider(self, response):
                item = BitcoinSpiderItem()
                item['article_name'] = response.request.meta['the_defiant_spider_title']
                articles = response.xpath('(//*[@id="content"]/div/div/div/section[1]/div/div/div[2]/div/div/div[7]/div)')
                # for article in articles:
                item['article_text'] = articles.xpath(".//p//text()").getall()
                item['article_text']=[i.replace("\t", "").replace("\n", "") for i in item['article_text']]
                item['article_text']=[i.replace("\u00A0", " ") for i in item['article_text']]


                yield item
                logging.info(response.url)
        else:
            def parse(self, response):
                the_defiant_spider = response.xpath('//*[@id="uc_post_list_elementor13294"]/div[1]/div[2]/div/div[1]')
                # // *[ @ id = "uc_post_list_elementor13294"] / div[1] / div[2] / div / div[1] / a

                # for bitc in bloomberg_spider:
                title = the_defiant_spider.xpath(".//a//text()").get()
                link = the_defiant_spider.xpath(".//a//@href").get()
                yield {"link": link, "title": title}
                yield {"meta": response}
                # yield response.follow(url=link, callback=self.parse_the_defiant_spider, meta={'the_defiant_spider_title': title})


            def parse_the_defiant_spider(self, response):
                item = BitcoinSpiderItem()
                item['article_name'] = response.request.meta['the_defiant_spider_title']
                articles = response.xpath('(//*[@id="content"]/div/div/div/section[1]/div/div/div[2]/div/div/div[7]/div)')
                # for article in articles:
                item['article_text'] = articles.xpath(".//p//text()").getall()
                item['article_text']=[i.replace("\t", "").replace("\n", "") for i in item['article_text']]
                item['article_text']=[i.replace("\u00A0", " ") for i in item['article_text']]


                yield item
                logging.info(response.url)
