import scrapy
import logging
from ..items import BitcoinSpiderItem
import os.path
from pathlib import Path
import sys



class TheBlockSpider(scrapy.Spider):
    # Name of the spider as mentioned in the "genspider" command
    name = 'the_block_spider'
    # Domains allowed for scraping, as mentioned in the "genspider" command
    allowed_domains = ['www.theblockcrypto.com', 'theblockcrypto.com']
    # URL(s) to scrape as mentioned in the "genspider" command
    # The scrapy spider, starts making  requests, to URLs mentioned here
    start_urls = ['https://www.theblockcrypto.com/category/bitcoin', 'https://www.theblockcrypto.com/category/ethereum']
    custom_settings = {"FEEDS": {"Scraper/Scraper/spiders/Scraped_data.json": {"format": "jsonlines"}}, 'FEED_EXPORTERS': {
            'jsonlines': 'scrapy.exporters.JsonItemExporter'}, 'FEED_EXPORT_ENCODING': 'utf-8'}

    for i in range(len(start_urls)):
        if i == 0:
            def parse(self, response):
                the_block_spider = response.xpath('//div[@class="feedCard-story"]/a[@class="theme color-outer-space"]')
                # for bitc in the_block_spider:
                title = the_block_spider.xpath(".//h3//text()").get()
                link = the_block_spider.xpath(".//@href").getall()
                # a[@class='theme color-outer-space']//
                # // *[ @ id = "__layout"] / div / div[3] / div[2] / div / div / div / article[1] / div[1] / div[2] / a[1]
                yield {"link": link}
                yield {"meta": response}
                # yield response.follow(url=link, callback=self.parse_the_block_spider, meta={'the_block_spider_title': title})


            def parse_the_block_spider(self, response):
                item = BitcoinSpiderItem()
                item['article_name'] = response.request.meta['the_block_spider_title']
                articles = response.xpath('(//*[@id="__layout"]/div/div[3]/div[1]/article/div/div/div[2]/div[3]/div[3])') #[9:33]

                item['article_text'] = articles.xpath(".//p//text()").getall()
                item['article_text']=[i.replace("\t", "").replace("\n", "") for i in item['article_text']]
                item['article_text']=[i.replace("\u00A0", " ") for i in item['article_text']]


                yield item
                logging.info(response.url)

        if i !=0:
            def parse(self, response):
                the_block_spider = response.xpath(
                    '//*[@id="__layout"]/div/div[3]/div[2]/div/div/div/article[1]/div[1]/div/h3/a')
                # for bitc in the_block_spider:
                title = the_block_spider.xpath(".//a[1]/h3/text()").get()
                link = the_block_spider.xpath(".//a/@href").get()
                yield {"link": link}
                yield {"meta": response}
                # yield response.follow(url=link, callback=self.parse_the_block_spider,
                #                       meta={'the_block_spider_title': title})

            def parse_the_block_spider(self, response):
                item = BitcoinSpiderItem()
                item['article_name'] = response.request.meta['the_block_spider_title']
                articles = response.xpath(
                    '(//*[@id="__layout"]/div/div[3]/div[1]/article/div/div/div[2]/div[2]/div[2])')

                item['article_text'] = articles.xpath(".//p//text()").getall()
                item['article_text'] = [i.replace("\t", "").replace("\n", "") for i in item['article_text']]
                item['article_text'] = [i.replace("\u00A0", " ") for i in item['article_text']]

                yield item
                logging.info(response.url)