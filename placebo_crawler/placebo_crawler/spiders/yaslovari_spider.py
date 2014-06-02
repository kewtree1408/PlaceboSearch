# coding: utf-8

from scrapy.contrib.spiders import CrawlSpider, Rule
from scrapy.contrib.linkextractors.sgml import SgmlLinkExtractor
from scrapy.selector import Selector
from scrapy.spider import Spider
from scrapy.http import Request
from placebo_crawler.items import DrugDescription, DiseaseDescription
# from scrapy import log
from html2text import html2text 

import time


class DrugsSpider(Spider):
    """
    Crawler for http://slovari.yandex.ru/~книги/РЛС/
    """

    name = 'yaslovari_drugs'
    allowed_domains = ['slovari.yandex.ru']
    # Регистр лекарственных средств России
    start_urls = ['http://slovari.yandex.ru/~книги/РЛС/']

    def p_between_id(self, n, sel):
        lineN = '//a[@class="article-subheader"][%s]/following-sibling::p/text()'%str(n)
        N = sel.xpath(lineN).extract()
        lineN_1 = '//a[@class="article-subheader"][%s]/preceding-sibling::p/text()'%str(n+1)
        N_1 = sel.xpath(lineN_1).extract()
        return [text for text in (set(N) & set(N_1))]

    def parse_drug(self, response):
        sel = Selector(response)
        drug_name = ''.join(sel.xpath('//h2[@class="b-serp__title"]/text()').extract()[0])
        # сохраняем всю информацию для индекса
        context = sel.xpath('//div[@class="body article"]').extract()[0]

        # сохраняем всю информацию для сниппета (пока не делаем классификацию, выяснить, нужно ли делать?)
        classification = ''

        all_subheads = sel.xpath('//a[@class="article-subheader"]/text()').extract()
        print all_subheads
        description, usage, contra, side, overdose = '', '', '', '', ''
        for i, subhead in enumerate(all_subheads):
            print i, subhead
            n = i+1
            if subhead == u"Состав и форма выпуска":
                description = ''.join(self.p_between_id(n, sel))
            elif subhead == u"Способ применения и дозы" or subhead == u"Показания":
                usage = ''.join(self.p_between_id(n, sel))
            elif subhead == u"Противопоказания":
                contra = ''.join(self.p_between_id(n, sel))
            elif subhead == u"Побочные действия":
                side = ''.join(self.p_between_id(n, sel))
            elif subhead == u"Передозировка":
                overdose = ''.join(self.p_between_id(n, sel))

        yield DrugDescription(  url=response.url,
                                name=drug_name,
                                classification=classification,
                                description=description,
                                usage=usage,
                                contra=contra,
                                side=side,
                                overdose=overdose,
                                info=html2text(context),
                            )

    def parse_letter(self, response):
        sel = Selector(response)
        url_drugs = sel.xpath('//h3[@class="b-serp-item__title"]/a/@href').extract()
        for url in url_drugs:
            root_url = 'http://slovari.yandex.ru' + url
            yield Request(root_url, callback=self.parse_drug)

    def parse(self, response):
        sel = Selector(response)
        url_letters = sel.xpath('//div[@class="b-book-info__index"]/a/@href').extract()
        p = 0
        for url in url_letters:
            root_url = 'http://slovari.yandex.ru' + url
            print root_url
            yield Request(root_url, callback=self.parse_letter)
            p += 1
            if p > 2:
                break
