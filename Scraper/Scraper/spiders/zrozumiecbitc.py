import scrapy
import logging
from ..items import ZrozumiecbitcSpiderItem


class ZrozumiecbitcSpider(scrapy.Spider):

    # Name of the spider as mentioned in the "genspider" command
    name = 'zrozumiecbitc'
    # Domains allowed for scraping, as mentioned in the "genspider" command
    allowed_domains = ['www.zrozumiecbitcoina.pl']
    # URL(s) to scrape as mentioned in the "genspider" command
    # The scrapy spider, starts making  requests, to URLs mentioned here
    start_urls = ['https://www.zrozumiecbitcoina.pl/2020/']

    # Default callback method responsible for returning the scraped output and processing it.
    def parse(self, response):
        # XPath expression of all the Quote elements.
        # All quotes belong to CSS attribute class having value 'quote'
        zrozumiecbitc = response.xpath("//h1/a")
        for bitc in zrozumiecbitc:
            title = zrozumiecbitc.xpath(".//text()").get()
            link = zrozumiecbitc.xpath(".//@href").get()

        yield response.follow(url=link, callback=self.parse_zrozumiecbitc, meta={'zrozumiecbitc_title': title})

    def parse_zrozumiecbitc(self, response):


        item = ZrozumiecbitcSpiderItem()
        item['article_name_1'] = response.request.meta['zrozumiecbitc_title']
        articles = response.xpath('(//*[@id="content-2403"])/div')
        # for article in articles:
        item['article_text_1'] = articles.xpath(".//p/text()").getall()

        # item['article_text_1'] = list(item['article_text_1'])
        # item['article_text_1'] = [i.replace("\t", "").replace("\n", "") for i in item['article_name_1']]
        # item['article_text_1'] = [i.replace("\u00A0", " ") for i in item['article_name_1']]
        yield item
        logging.info(response.url)








