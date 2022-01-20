import scrapy
import logging
from ..items import ZrozumiecbitcSpiderItem
import os.path
from pathlib import Path
import sys

path = Path("C:/Users/Jacklord/PycharmProjects/Crypto_nlp/Crypto_nlp/Scraper/Scraper/spiders/Scraped_data_no_quotes1.json")

if os.path.exists(path) is True:
    print("file is already changed, updated data in BQ")
    from ..create_table_ndjson_bq import dataload
    dataload()

else:
    if os.path.isfile('C:/Users/Jacklord/PycharmProjects/Crypto_nlp/Crypto_nlp/Scraper/Scraper/spiders/Scraped_data.json'):
        from ..custom_text_edit import customtextfunc
        from ..create_table_ndjson_bq import dataload
        # from ..create_table_ndjson_bq import dataload
        file = open('C:/Users/Jacklord/PycharmProjects/Crypto_nlp/Crypto_nlp/Scraper/Scraper/spiders/Scraped_data.json',
                    "r")
        customtextfunc(file)
        dataload()

    else:
        class ZrozumiecbitcSpider(scrapy.Spider):
            name = 'zrozumiecbitc'
            allowed_domains = ['www.zrozumiecbitcoina.pl']
            start_urls = ['https://www.zrozumiecbitcoina.pl/2020/']

            def parse(self, response):
                zrozumiecbitc = response.xpath("//h1/a")
                for bitc in zrozumiecbitc:
                    title = zrozumiecbitc.xpath(".//text()").get()
                    link = zrozumiecbitc.xpath(".//@href").get()
                # yield {
                #     'bitc_art_text': title,
                #     'bitc_link': link
                # }
                yield response.follow(url=link, callback=self.parse_zrozumiecbitc, meta={'zrozumiecbitc_title': title})

            def parse_zrozumiecbitc(self, response):
                item = ZrozumiecbitcSpiderItem()
                item['article_name'] = response.request.meta['zrozumiecbitc_title']
                articles = response.xpath('(//*[@id="content-2403"])/div')
                # for article in articles:
                item['text'] = articles.xpath(".//p/text()").getall()
                yield item

                logging.info(response.url)

                from ..custom_text_edit import customtextfunc
                file = open(
                    'C:/Users/Jacklord/PycharmProjects/Crypto_nlp/Crypto_nlp/Scraper/Scraper/spiders/Scraped_data.json',
                    "r")
                customtextfunc(file)





