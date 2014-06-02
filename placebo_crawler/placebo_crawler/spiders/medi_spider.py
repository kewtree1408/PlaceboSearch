# coding: utf-8

from scrapy.selector import Selector
from scrapy.spider import Spider
from scrapy.http import Request
from placebo_crawler.items import DrugDescription
# from scrapy import log
from html2text import html2text 


class DrugsSpider(Spider):
    """
    Crawler for http://medi.ru/
    """

    name = 'medi_drugs'
    allowed_domains = ['medi.ru']
    start_urls = ['http://medi.ru/']

    def parse_drug(self, response):
        sel = Selector(response)
        drug_name = sel.xpath('//h1[@itemprop="name"]/text()').extract()
        if drug_name:
            drug_name = drug_name[0]
            context = sel.xpath('//div[@itemtype="http://schema.org/Drug"]').extract()[0]
            
            classification, description, usage, contra, side, overdose = '', '', '', '', '', ''

            all_subheads = sel.xpath('//div[@itemtype="http://schema.org/Drug"]//text()').extract()

            for i in reversed(range(len(all_subheads))):
                s = all_subheads[i].replace(':', '').replace('.', '').lower()
                if s == u"передозировка":
                    overdose = ''.join(all_subheads[i:])
                    all_subheads = all_subheads[:i]
                elif s == u"побочное действие" or s.startswith(u"побочные"):
                    side = ''.join(all_subheads[i:])
                    all_subheads = all_subheads[:i]
                elif s == u"противопоказания":
                    contra = ''.join(all_subheads[i:])
                    all_subheads = all_subheads[:i]
                elif s.startswith(u"показания"):
                    usage = ''.join(all_subheads[i:])
                    all_subheads = all_subheads[:i]
                elif s == u"состав":
                    description = ''.join(all_subheads[i:])
                    all_subheads = all_subheads[:i]
                
            yield DrugDescription(url = response.url,
                                  name = drug_name,
                                  classification = classification,
                                  description = description,
                                  usage = usage,
                                  contra = contra,
                                  side = side,
                                  overdose = overdose,
                                  info = html2text(context),)

    def parse_letter(self, response):
        sel = Selector(response)
        paragraphs = sel.xpath('//ul/p')
        for par in paragraphs:
            s = par.xpath('a/text()').extract()
            if s:
                if s[0] == u'Официальная инструкция':
                    drug_url = par.xpath('a/@href').extract()
                    if drug_url:
                        drug_url = 'http://medi.ru/doc/' + drug_url[0]
                        yield Request(drug_url, callback=self.parse_drug)

    def parse(self, response):
        sel = Selector(response)
        url_letters = sel.xpath('//nobr/a/@href').extract()
        for url in url_letters:
            root_url = 'http://medi.ru/' + url[2:]
            yield Request(root_url, callback=self.parse_letter)

