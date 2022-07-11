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
    start_urls = ['https://www.zrozumiecbitcoina.pl']

    # Default callback method responsible for returning the scraped output and processing it.
    def parse(self, response):
        # XPath expression of all the Quote elements.
        # All quotes belong to CSS attribute class having value 'quote'
        # base_url = 'https://www.zrozumiecbitcoina.pl/2020'
        # zrozumiecbitc = response.xpath("//h1/a")
        zrozumiecbitc = response.xpath("/html/body/div[2]/div/div/section[8]/div/div/div/div/div/div[1]/div/div/article[4]/div/div[3]/h3/a")

        # for bitc in zrozumiecbitc:
        title = zrozumiecbitc.xpath(".//text()").get()
        link = zrozumiecbitc.xpath(".//@href").get()
        # full_link=base_url+link

        yield response.follow(url=link, callback=self.parse_zrozumiecbitc, meta={'zrozumiecbitc_title': title, 'zrozumiecbitc_link': link})

    def parse_zrozumiecbitc(self, response):
        item = ZrozumiecbitcSpiderItem()
        item['article_name'] = response.request.meta['zrozumiecbitc_title']
        item['article_link'] =response.request.meta['zrozumiecbitc_link']

        articles = response.xpath('(//*[@id="content-3476"]/div[2])')
        # for article in articles:
        item['article_text'] = articles.xpath(".//p/text()").getall()

        yield item
        logging.info(response.url)








