import scrapy
import logging
from ..items import ZrozumiecbitcSpiderItem

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
        item['art_title'] = response.request.meta['zrozumiecbitc_title']
        articles = response.xpath('(//*[@id="content-2403"])/div')
        for article in articles:
            item['article_text'] = article.xpath(".//p/text()").getall()
            yield item
                # 'article title': item


        logging.info(response.url)


# //*[@id="content-2403"]/header/div[@class="block-double"]/h1/a


# '//*[@id="content-2403"]/div["content-item-test content-item inner block-double"]/p'
