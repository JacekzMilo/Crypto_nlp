import scrapy
import logging
from ..items import BitcoinSpiderItem
import os.path
from pathlib import Path
import sys



class CoinDeskSpider(scrapy.Spider):
    # Name of the spider as mentioned in the "genspider" command
    name = 'coin_desk_spider'
    # Domains allowed for scraping, as mentioned in the "genspider" command
    allowed_domains = ['www.coindesk.com', 'coindesk.com']
    # URL(s) to scrape as mentioned in the "genspider" command
    # The scrapy spider, starts making  requests, to URLs mentioned here
    start_urls = ['https://www.coindesk.com/markets', 'https://www.coindesk.com/tech/']
    custom_settings = {"FEEDS": {"Scraper/Scraper/spiders/Scraped_data.json": {"format": "jsonlines"}}, 'FEED_EXPORTERS': {
            'jsonlines': 'scrapy.exporters.JsonItemExporter'}, 'FEED_EXPORT_ENCODING': 'utf-8'}

    for i in range(len(start_urls)):
        if i == 0:
            def parse(self, response):
                base_url = 'https://www.coindesk.com'
                coin_desk_spider = response.xpath('//*[@id="fusion-app"]/div/div[2]/div/main/section[1]/div/div[1]/div[2]/div[1]/div[1]/div/div[2]/h3/a')
                for bitc in coin_desk_spider:
                    title = coin_desk_spider.xpath(".//text()").get()
                    link = coin_desk_spider.xpath(".//@href").get()
                    full_url = base_url + link
                # yield {"link": link, "title": title}
                # yield {"meta": response}
                yield response.follow(url=link, callback=self.parse_coin_desk_spider, meta={'coin_desk_spider_title': title, 'coin_desk_spider_link': full_url})


            def parse_coin_desk_spider(self, response):
                item = BitcoinSpiderItem()
                item['article_name'] = response.request.meta['coin_desk_spider_title']
                item['article_link'] = response.request.meta['coin_desk_spider_link']

                articles = response.xpath('(//*[@id="fusion-app"]/div/div[2]/main/div[1]/div/section/div/div[3]/div[1]/div)') #[9:33]

                item['article_text'] = articles.xpath(".//div/*//text()").getall()
                item['article_text']=[i.replace("\t", "").replace("\n", "") for i in item['article_text']]
                item['article_text']=[i.replace("\u00A0", " ") for i in item['article_text']]


                yield item
                logging.info(response.url)

        if i !=0:
            def parse(self, response):
                base_url = 'https://www.coindesk.com'
                coin_desk_spider = response.xpath(
                    '//*[@id="fusion-app"]/div/div[2]/div/main/section[1]/div/div[1]/div[2]/div[1]/div[1]/div/div[2]/h3/a')
                # for bitc in coin_desk_spider:
                title = coin_desk_spider.xpath(".//text()").get()
                link = coin_desk_spider.xpath(".//@href").get()
                full_url = base_url + link

                # yield {"link": link, "title": title}
                # yield {"meta": response}
                yield response.follow(url=link, callback=self.parse_coin_desk_spider,
                                      meta={'coin_desk_spider_title': title, 'coin_desk_spider_link': full_url})

            def parse_coin_desk_spider(self, response):
                item = BitcoinSpiderItem()
                item['article_name'] = response.request.meta['coin_desk_spider_title']
                item['article_link'] = response.request.meta['coin_desk_spider_link']

                articles = response.xpath(
                    '(//*[@id="fusion-app"]/div/div[2]/main/div[1]/div/section/div/div[3]/div[1]/div)')

                item['article_text'] = articles.xpath(".//div/p//text()").getall()
                item['article_text'] = [i.replace("\t", "").replace("\n", "") for i in item['article_text']]
                item['article_text'] = [i.replace("\u00A0", " ") for i in item['article_text']]

                yield item
                logging.info(response.url)
