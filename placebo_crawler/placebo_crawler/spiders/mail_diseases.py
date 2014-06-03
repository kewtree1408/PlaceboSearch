# coding: utf-8

from scrapy.selector import Selector
from scrapy.spider import Spider
from scrapy.http import Request
from placebo_crawler.items import DiseaseDescription
# from scrapy import log
from html2text import html2text 

import time

class DiseaseSpider(Spider):
    name = 'mailru_diseases'
    allowed_domains = ['mail.ru']
    start_urls = ['http://health.mail.ru/disease/']

    def parse_disease(self, response):
        sel = Selector(response)
        disease_title = sel.xpath('//h1[@class="page-info__title"]/text()').extract()[0]
        context = ' '.join(sel.xpath('//div[@class="column__air"]/div//text()').extract())        
        yield  DiseaseDescription(url = response.url,
                                 name = disease_title,
                                 description = html2text(context),
                                 drugs = "",)

    def parse_rubric(self, response):
        sel = Selector(response)
        url_diseases = sel.xpath('//a[@class="list__title link-holder"]/@href').extract()
        for url in url_diseases:
            url = 'http://health.mail.ru/disease/' + url[9:]
            yield Request(url, callback=self.parse_disease)
            #time.sleep(5)

    def parse(self, response):
        sel = Selector(response)
        url_rubrics = sel.xpath('//div[@class="catalog__rubric"]/a/@href').extract()
        print 'Hello'
        for url in url_rubrics:
            url = 'http://health.mail.ru/disease/' + url[9:]
            yield Request(url, callback=self.parse_rubric)
            #time.sleep(5)
