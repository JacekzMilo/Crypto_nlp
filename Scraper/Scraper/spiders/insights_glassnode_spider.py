import scrapy
import logging
from ..items import BitcoinSpiderItem, InsightsGlassnodeItem
import os.path
from pathlib import Path
import sys



class InsightsGlassnodeSpider(scrapy.Spider):
    # Name of the spider as mentioned in the "genspider" command
    name = 'insights_glassnode_spider'
    # Domains allowed for scraping, as mentioned in the "genspider" command
    allowed_domains = ['www.insights.glassnode.com', 'insights.glassnode.com']
    # URL(s) to scrape as mentioned in the "genspider" command
    # The scrapy spider, starts making  requests, to URLs mentioned here
    start_urls = ['https://insights.glassnode.com/']
    custom_settings = {"FEEDS": {"Scraper/Scraper/spiders/Scraped_data.json": {"format": "jsonlines"}}, 'FEED_EXPORTERS': {
            'jsonlines': 'scrapy.exporters.JsonItemExporter'}, 'FEED_EXPORT_ENCODING': 'utf-8'}

    def parse(self, response):
        insights_glassnode_spider = response.xpath('//*[@id="site-main"]/div/div/article[1]/div/a')
        for bitc in insights_glassnode_spider:
            title = insights_glassnode_spider.xpath(".//header/h2/text()").get()
            link = insights_glassnode_spider.xpath(".//@href").get()
        # yield {"link": link, "title": title}
        # yield {"meta": response}
        yield response.follow(url=link, callback=self.parse_insights_glassnode_spider, meta={'insights_glassnode_spider_title': title})


    def parse_insights_glassnode_spider(self, response):
        item = InsightsGlassnodeItem()
        item['article_name'] = response.request.meta['insights_glassnode_spider_title']
        articles = response.xpath('(//*[@id="site-main"]/article)')
        # for article in articles:
        item['article_text'] = articles.xpath(".//header/p/text()").getall()
        item['article_text'] = item['article_text']+articles.xpath(".//section/p//text()").getall()

        item['article_text']=[i.replace("\t", "").replace("\n", "") for i in item['article_text']]
        item['article_text']=[i.replace("\u00A0", " ") for i in item['article_text']]

        # item['article_text_2'] = list(item['article_text_2'])
        # item['article_text'] = [i.replace("\", " ") for i in item['article_text']]
        yield item
        logging.info(response.url)