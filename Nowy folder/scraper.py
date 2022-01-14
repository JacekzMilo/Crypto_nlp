# Import scrapy modules
from scrapy.crawler import CrawlerProcess
from scrapy import settings


from common.spiders.spider1 import FirstSpider
# from common.spiders.spider2 import SecondSpider
# from common.spiders.spider3 import ThirdSpider


# Initiate a Crawling process
process = CrawlerProcess()



# Tell the scraper which spider to use
process.crawl(FirstSpider)
# process.crawl(SecondSpider)
# process.crawl(ThirdSpider)

# Start the crawling
process.start()