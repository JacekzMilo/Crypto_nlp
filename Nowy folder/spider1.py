import scrapy
import re
import json
import os


class FirstSpider(scrapy.Spider):
    name = "spider1"
    allowed_domains = ['www.zrozumiecbitcoina.pl']


# Initialize requests
def start_requests(self):
    # List of URL to request
    urls = ['https://www.zrozumiecbitcoina.pl/2020/08/06/zyskowny-lipiec-defi-na-calego-raport-z-portfela-za-lipiec-2020/']
    for url in urls:
        # We use yield and use parse as a method to parse
        # the information
        yield scrapy.Request(url=url, callback=self.parse)



def parse(self, response):
    links = response.xpath('//div[@class="content-item-text content-item inner block-double"]/p[*]').extract()
    for link in links:
        yield response.follow(url=link, callback=self.parse_first)


# Formating date as YYYY-MM-DD HH:MM:SS
def extractdate(self, text):
    date = re.search("(\d{2}/\d{2}/\d{4})$", text).group(1)
    return date


def parse_first(self, response):
    text = []
    title = response.xpath("//*[@id='content-2063']/header/div[2]/h1").get()
    for paragraph in response.xpath("//*[@id='bodyDocument']/p"):
        text.append(paragraph.get())


    # we join all the elements of the list together
    text = " ".join(text)

    # We extract the date
    date = self.extractdate(text)

    document = {
        "date": date,
        "title": title,
        "text": text
    }




    json_file = "./json/documents.json"
    with open(json_file) as file:
        data = json.load(file, encoding='utf-8')

    if type(data) is dict:
        data = [data]

    data.append(document)

    with open(json_file, 'w') as file:
        json.dump(data, file, ensure_ascii=False)