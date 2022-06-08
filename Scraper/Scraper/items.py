# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class ZrozumiecbitcSpiderItem(scrapy.Item):
    # define the fields for your item here like:
    article_text_1 = scrapy.Field()
    article_name_1 = scrapy.Field()
    # title = scrapy.Field()

class BitcoinSpiderItem(scrapy.Item):
    article_text_2 = scrapy.Field()
    article_name_2 = scrapy.Field()
    # title = scrapy.Field()
