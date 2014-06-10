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
    Crawler for http://www.wiki-meds.ru/lekarstvennie-preparati-alpha
    """

    name = 'wikimeds_drugs'
    allowed_domains = ['wiki-meds.ru']
    #download_delay = 2
    #http_user = 'crawler'
    #http_pass = '123'
    # Раздел "Алфавитный указатель"
    start_urls = ['http://www.wiki-meds.ru/lekarstvennie-preparati-alpha']


    def p_between_id(self, n, sel):
        #lineN = '//div[@id="printMe"]//h2[%s]/following-sibling::*[self::p or self::span/p or self::span/table or self::ul/li]//text()'%str(n)
        lineN = '//div[@id="printMe"]//h2[%s]/following-sibling::*[self::p or self::span or self::ul]//text()'%str(n)
        N = sel.xpath(lineN).extract()
        #lineN_1 = '//div[@id="printMe"]//h2[%s]/preceding-sibling::*[self::p or self::span/p or self::span/table or self::ul/li]//text()'%str(n+1)
        lineN_1 = '//div[@id="printMe"]//h2[%s]/preceding-sibling::*[self::p or self::span or self::ul]//text()'%str(n+1)
        N_1 = sel.xpath(lineN_1).extract()

        text = []
        for string_n in N:
            for string_n_1 in N_1:
                if string_n == string_n_1:
                    if not string_n == '\n': # часто переводы строк
                        text.append(string_n)

        return text


    def parse_drug(self, response):
        sel = Selector(response)
        drug_name = sel.xpath('//div[@id="printMe"]/h1/text()').extract()[0]
        context = sel.xpath('//div[@id="printMe"]').extract()[0]
        classification = ''

        all_subheads = sel.xpath('//div[@id="printMe"]/h2/text()').extract()
        description, usage, contra, side, overdose = '', '', '', '', ''
        for i, subhead in enumerate(all_subheads):
            n = i+1
            if u"Лекарственная форма, состав, упаковка" in subhead:
                description = ''.join(self.p_between_id(n, sel))
            elif u"Режим дозирования" in subhead or u"Показания к применению" in subhead:
                usage = ''.join(self.p_between_id(n, sel))
            elif u"Противопоказания" in subhead:
                contra = ''.join(self.p_between_id(n, sel))
            elif u"Побочные действия" in subhead:
                side = ''.join(self.p_between_id(n, sel))
            elif u"Передозировка при приёме" in subhead:
                overdose = ''.join(self.p_between_id(n, sel))
        #time.sleep(5)
        yield DrugDescription(
                        url=response.url,
                        name=drug_name,
                        classification=classification,
                        description=description,
                        usage=usage,
                        contra=contra,
                        side=side,
                        overdose=overdose,
                        info=html2text(context),
                    )



    def parse_page(self, response):
        sel = Selector(response)
        url_drugs = sel.xpath('//div[@id="printMe"]/div[@class="drugInfoContainer"]/a/@href').extract()
        for url in url_drugs:
            request = Request(url, callback=self.parse_drug)
            #request.meta['url'] = url
            print url
            yield request

    def parse_letter(self, response):
        """
        target_url = response.meta['url']
        if(not target_url == response.url):
            request = Request(target_url, callback=self.parse_letter)
            request.meta['url'] = target_url
            yield request
        """
        #if not response.status == 200:
        #    1+1
        sel = Selector(response)
        #last_page = sel.xpath('//tbody//td[@class="right bottom"]/a/@href').extract()
        #if last_page:
            #last_page = last_page[0]
        chr_number = response.url.rindex('/')
        for i in range(1, 200):
            url = response.url[:chr_number+1] + str(i)
            request = Request(url, callback=self.parse_page)
            print "  next page " + url
            #request.meta['url'] = url
            #time.sleep(5)
            yield request
        #else:
        #    self.parse_page(response)




    def parse(self, response):
        sel = Selector(response)
        url_letters = sel.xpath('//div/ul[@class="alphaLinks"]//a/@href').extract()
        for url in url_letters:
            print url
            #time.sleep(5)
            yield Request(url, callback=self.parse_letter)







# Не скачивает названия лекарств (там отдельная страница со списком)
class DiseaseSpider(Spider):
    """
    Crawler for http://www.wiki-meds.ru/zabolevaniya-alpha
    """

    name = 'wikimeds_diseases'
    allowed_domains = ['wiki-meds.ru']
    #download_delay = 2
    #handle_httpstatus_list = [302]
    #http_user = 'crawler'
    #http_pass = '123'
    # Раздел "Алфавитный указатель болезней"
    start_urls = ['http://www.wiki-meds.ru/zabolevaniya-alpha']

    def parse_disease(self, response):
        """
        target_url = response.meta['url']
        if(not target_url == response.url):
            request = Request(target_url, callback=self.parse_disease)
            request.meta['url'] = target_url
            yield request
        """
        #if not response.status == 200:
        #    1+1
        sel = Selector(response)
        disease_title = sel.xpath('//div[@id="printMe"]/h1/text()').extract()[0]
        # context = sel.xpath('//*[@id="div_nest"]').extract()[0]q

        contexte = ''
        temp = ''.join([stuff+'\n' for stuff in sel.xpath('//div[@id="printMe"]//span[@id="info576"]/*[self::p or self::ol or self::ul]//text()').extract()])
        if temp:
            context = html2text(temp)
            drugs_and_stuff = ''.join([stuff+'\n' for stuff in sel.xpath('//a[@class="rest_data_list"]/text()').extract()])
            #time.sleep(5)
            yield DiseaseDescription(   url = response.url,
                                        name = disease_title,
                                        description = context,
                                        drugs = drugs_and_stuff,
                                    )

    def parse_page(self, response):
        sel = Selector(response)
        url_drugs = sel.xpath('//div[@id="printMe"]/div[@class="illnessInfoContainer"]/p[@align="left"]/a[@class="drugNameLink"]/@href').extract()
        for url in url_drugs:
            request = Request(url, callback=self.parse_disease)
            #request.meta['url'] = url
            print url
            yield request

    def parse_letter(self, response):
        """
        target_url = response.meta['url']
        if(not target_url == response.url):
            request = Request(target_url, callback=self.parse_letter)
            request.meta['url'] = target_url
            yield request
        """
        #if not response.status == 200:
        #    1+1
        sel = Selector(response)
        #last_page = sel.xpath('//tbody//td[@class="right bottom"]/a/@href').extract()
        #if last_page:
            #last_page = last_page[0]
        chr_number = response.url.rindex('/')
        for i in range(1, 200):
            url = response.url[:chr_number+1] + str(i)
            request = Request(url, callback=self.parse_page)
            print "  next page " + url
            #request.meta['url'] = url
            #time.sleep(5)
            yield request
        #else:
        #    self.parse_page(response)


    def parse(self, response):
        #if not response.status == 200:
        #    1+1
        sel = Selector(response)
        url_letters = sel.xpath('//div/ul[@class="alphaLinks"]//a/@href').extract()
        for url in url_letters:
            print url
            request = Request(url, callback=self.parse_letter)
            #request.meta['url'] = url
            #time.sleep(5)
            yield request
