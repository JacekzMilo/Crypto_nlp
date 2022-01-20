# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class ZrozumiecbitcSpiderItem(scrapy.Item):
    # define the fields for your item here like:
    text = scrapy.Field()
    article_name = scrapy.Field()
    title = scrapy.Field()
