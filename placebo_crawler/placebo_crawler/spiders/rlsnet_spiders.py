# coding: utf-8

from scrapy.contrib.spiders import CrawlSpider, Rule
from scrapy.contrib.linkextractors.sgml import SgmlLinkExtractor
from scrapy.selector import Selector
from scrapy.spider import Spider
from scrapy.http import Request
from placebo_crawler.items import DrugDescription, DrugInfo, DiseaseDescription
# from scrapy import log
from html2text import html2text 

import time


class DrugsSpider(Spider):
    """
    Crawler for http://www.rlsnet.ru
    """

    name = 'rlsnet_drugs'
    allowed_domains = ['rlsnet.ru']
    # Раздел "Алфавитный указатель"
    start_urls = ['http://www.rlsnet.ru']

    # выдираем p-ку строго после заданного заголовка (скорее всего работает не для всех страниц)
    # todo: переделать так, чтобы работало для всех страниц
    def p_between_h(self, number_hN, sel):
        # lineN = '//h2[%s]/following-sibling::p/text()'%str(number_hN)
        lineN = '//h2[%s]/following-sibling::p/text()'%str(number_hN)
        hN = sel.xpath(lineN).extract()
        # lineN_1 = '//h2[%s]/preceding-sibling::p/text()'%str(number_hN+1)
        lineN_1 = '//h2[%s]/preceding-sibling::p/text()'%str(number_hN+1)
        hN_1 = sel.xpath(lineN_1).extract()
        return [text for text in (set(hN) & set(hN_1))]

    def parse_disease(self, response):
        sel = Selector(response)
        disease_title = sel.xpath('//h1[@id="page_head"]/text()').extract()[0]
        # context = sel.xpath('//*[@id="div_nest"]').extract()[0]
        context = sel.xpath('//*[@class="news_text full"]').extract()[0]
        drugs_and_stuff = ''.join([stuff+'\n' for stuff in sel.xpath('//a[@class="rest_data_list"]/text()').extract()])
        yield DiseaseDescription(   url = response.url,
                                    name = disease_title, 
                                    description = html2text(context), 
                                    drugs = drugs_and_stuff,
                                )

    def parse_drug(self, response):
        sel = Selector(response)
        drug_name = sel.xpath('//span[@part="rusname"]/text()').extract()[0]
        context = sel.xpath('//*[@id="tn_content"]').extract()[0]
        # сохраняем всю информацию для индекса
        time.sleep(1)
        yield DrugInfo(url=response.url, name=drug_name, info=html2text(context))

        # сохраняем всю информацию для сниппета (пока не делаем, выяснить, нужно ли делать?)
        classification = ''
        for nozo_class in sel.xpath('//*[@class="field_20110303"]/li/a'):
            classification += nozo_class.xpath('text()').extract()[0] + '; '
            
        description = ''.join(self.p_between_h(6, sel))
        usage = ''.join(self.p_between_h(9, sel))
        contra = ''.join(self.p_between_h(10, sel))
        side = ''.join(self.p_between_h(11, sel))
        overdose = ''.join(self.p_between_h(12, sel))
        time.sleep(1)
        yield DrugDescription(  url = response.url,
                                name = drug_name,
                                classification = classification,
                                description = description,
                                usage = usage,
                                contra = contra,
                                side = side,
                                overdose = overdose,
                            )

    def parse_letter(self, response):
        sel = Selector(response)
        url_drugs = sel.xpath('//ul/li/a/@href').extract()
        for url in url_drugs:
            if "#" not in url:
                yield Request(url, callback=self.parse_drug)
                time.sleep(2)
                # break

    def parse(self, response):
        sel = Selector(response)
        url_letters = sel.xpath('//div[@class="tn_letters"]/a/@href').extract()
        for url in url_letters:
            yield Request(url, callback=self.parse_letter)
            time.sleep(2)
            # break


class DiseaseSpider(Spider):
    """
    Crawler for http://www.rlsnet.ru
    """

    name = 'rlsnet_diseases'
    allowed_domains = ['rlsnet.ru']
    # Раздел "Алфавитный указатель болезней"
    start_urls = ['http://www.rlsnet.ru/mkb_alf.htm']

    def parse_disease(self, response):
        sel = Selector(response)
        disease_title = sel.xpath('//h1[@id="page_head"]/text()').extract()[0]
        # context = sel.xpath('//*[@id="div_nest"]').extract()[0]
        context = sel.xpath('//*[@class="news_text full"]').extract()[0]
        drugs_and_stuff = ''.join([stuff+'\n' for stuff in sel.xpath('//a[@class="rest_data_list"]/text()').extract()])
        yield DiseaseDescription(   url = response.url,
                                    name = disease_title, 
                                    description = html2text(context), 
                                    drugs = drugs_and_stuff,
                                )

    def parse_letter(self, response):
        sel = Selector(response)
        url_diseases = sel.xpath('//ul/li/a/@href').extract()
        for url in url_diseases:
            yield Request(url, callback=self.parse_disease)
            time.sleep(2)
            # break

    def parse(self, response):
        sel = Selector(response)
        url_letters = sel.xpath('//th/a/@href').extract()
        i = 0
        for url in url_letters:
            yield Request(url, callback=self.parse_letter)
            time.sleep(2)
            # i += 1
            # if i>4:
            #     break



        
            
            

        
            
            
