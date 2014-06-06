# coding: utf-8

from scrapy.selector import Selector
from scrapy.spider import Spider
from scrapy.http import Request
from placebo_crawler.items import DrugSynonyms
# from scrapy import log

#import time


class DrugSynonymsSpider(Spider):
    name = 'drug_synonyms'
    allowed_domains = ['lekspr.ru']
    start_urls = ['http://www.lekspr.ru/']

    def parse_drug(self, response):
        sel = Selector(response)
        name = sel.xpath('//b[@class="ch_title_o"]/text()').extract()[0].lower()
        par = sel.xpath('//td[@align="left"]/p[2]/text()').extract()
        if par:
            par = par[0].lower()
            if par.startswith(u'синоним'):
                par = par.split(':')
                if len(par) > 1:
                    synonyms = par[1].split(',')
                    for i in range(len(synonyms)):
                        synonyms[i] = synonyms[i].strip().strip('.')
                    #synonyms = "|".join(synonyms)
                    yield DrugSynonyms(url=response.url, name=name, synonyms=synonyms)

    def parse_letter(self, response):
        sel = Selector(response)
        url_drugs = sel.xpath('//a[@class="ch_title_g"]/@href').extract()
        for url in url_drugs:
            url = 'http://www.lekspr.ru/' + url
            print url
            yield Request(url, callback=self.parse_drug)
            #time.sleep(5)

    def parse(self, response):
        for i in range(2, 28):
            url = 'http://www.lekspr.ru/' + 'sections_' + str(i) + '.html'
            print url
            yield Request(url, callback=self.parse_letter)
            #time.sleep(5)
