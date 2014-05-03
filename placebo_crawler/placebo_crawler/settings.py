# Scrapy settings for placebo_crawler project
#
# For simplicity, this file contains only the most important settings by
# default. All the other settings are documented here:
#
#     http://doc.scrapy.org/en/latest/topics/settings.html
#

BOT_NAME = 'placebo_crawler'

SPIDER_MODULES = ['placebo_crawler.spiders']
NEWSPIDER_MODULE = 'placebo_crawler.spiders'

ITEM_PIPELINES = {
    'placebo_crawler.pipelines.JsonWriterPipeline': 400,
    'placebo_crawler.pipelines.DuplicatesPipeline': 800,
}

ROBOTSTXT_OBEY = True

# Crawl responsibly by identifying yourself (and your website) on the user-agent
#USER_AGENT = 'placebo_crawler (+http://www.yourdomain.com)'
