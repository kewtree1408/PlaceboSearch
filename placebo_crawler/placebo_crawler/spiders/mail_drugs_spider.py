# coding: utf-8

from scrapy.contrib.spiders import CrawlSpider, Rule
from scrapy.contrib.linkextractors.sgml import SgmlLinkExtractor
from scrapy.selector import Selector
from scrapy.spider import Spider
from scrapy.http import Request
from placebo_crawler.items import DrugDescription, DiseaseDescription
from html2text import html2text

import time

class DrugsSpider(Spider):
    """
    Crawler for http://www.health.mail.ru
    """

    name = 'mailru_drugs'
    allowed_domains = ['health.mail.ru']
    # Раздел: "Лекарства группированы по направлению, на что они действуют
    # (пищеварительный тракт и обмен веществ, дерматология и прочее)"
    start_urls = ['http://health.mail.ru/drug/']
    name_domain = 'http://health.mail.ru' # название домена для относительных ссылок на сайте

    def p_between_id(self, n, sel):
        lineN = '//div[@class="text margin_bottom_30 js-text_widget"]//h2[%s]/following-sibling::*[self::p or self::table or self::div/p]//text()'%str(n)
        N = sel.xpath(lineN).extract()
        lineN_1 = '//div[@class="text margin_bottom_30 js-text_widget"]//h2[%s]/preceding-sibling::*[self::p or self::table or self::div/p]//text()'%str(n+1)
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
        drug_name = sel.xpath('//h1[@class="page-info__title"]/text()').extract()[0]
        context = sel.xpath('//div[@class="column__air"]').extract()[0]
        classification = ''

        all_subheads = sel.xpath('//div[@class="text margin_bottom_30 js-text_widget"]//h2/text()').extract()
        description, usage, contra, side, overdose = '', '', '', '', ''
        for i, subhead in enumerate(all_subheads):
            n = i+1
            if u"Форма выпуска, состав и упаковка" in subhead:
                description = ''.join(self.p_between_id(n, sel))
            elif u"Дозировка" in subhead or u"Показания" in subhead:
                usage = ''.join(self.p_between_id(n, sel))
            elif u"Противопоказания" in subhead:
                contra = ''.join(self.p_between_id(n, sel))
            elif u"Побочные действия" in subhead:
                side = ''.join(self.p_between_id(n, sel))
            elif u"Передозировка" in subhead:
                overdose = ''.join(self.p_between_id(n, sel))
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

    def parse_list_of_drugs(self, response):
        sel = Selector(response)
        url_drugs = sel.xpath('//a[@class="entry__link link-holder"]//@href').extract() # Вытаскиваем все ссылки с текущей страницы
        #print(url_drugs)
        for url in url_drugs:
            absolutely_url_drug = self.name_domain + url
            yield Request(absolutely_url_drug, callback=self.parse_drug)

    def parse_setof_pages(self, response):
        sel = Selector(response)
        all_another_pages = sel.xpath('//div[@class="paging"]//a[@class="paging__item"]/@href').extract() # Проверяем есть ли другие страницы
        #print response.url
        all_another_pages.append(response.url)
        for page_link in all_another_pages:
            absolutely_page_link = page_link
            if '?' in page_link:
                absolutely_page_link = self.name_domain + page_link
            yield Request(absolutely_page_link, callback=self.parse_list_of_drugs)

    def parse(self, response):
        sel = Selector(response)
        #print "\n\nstart\n\n"#
        rubric_links = sel.xpath('//div[@class="hidden hidden_small"]//div[@class="catalog__rubric"]//@href').extract() # получаем локальные ссылки на тематику к которым принадлежат лекарства (91шт)
        for link in rubric_links:
            catalog_item_link = self.name_domain + link
            yield Request(catalog_item_link, callback=self.parse_setof_pages)
            # return
