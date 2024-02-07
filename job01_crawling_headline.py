import pandas as pd
from bs4 import BeautifulSoup #pip install bs4
import requests #pip install requests
import re
import pandas as ps #pip install pandas
import datetime

##

category = ['Politics', 'Economic', 'Social', 'Culture', 'World', 'IT']

url = 'https://news.naver.com/main/main.naver?mode=LSD&mid=shm&sid1=100'
headers = {"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
#resp = requests.get(url, headers = headers)
#print(resp)
#print(type(resp))
#print(list(resp))
#soup = BeautifulSoup(resp.text, 'html.parser')
#print(soup)
#title_tags = soup.select('.sh_text_headline')
#print(title_tags)
#print(len(title_tags))
#print(type(title_tags[0]))
#titles = []
#for title_tags in title_tags:
   #titles.append(re.compile('[^가-힣|a-z|A-Z]').sub(' ', title_tags.text))
#print(titles)

df_fitles = pd.DataFrame()
re_title = re.compile('[^가-힣|a-z|A-Z]')

for i in range(6):
    url = 'https://news.naver.com/main/main.naver?mode=LSD&mid=shm&sid1=10{}'.format(i)
    resp = requests.get(url, headers = headers)
    soup = BeautifulSoup(resp.text, 'html.parser')
    title_tags = soup.select('.sh_text_headline')
    titles = []
    for title_tag in title_tags:
        titles.append(re_title.sub(' ', title_tag.text))
    df_section_titles = pd.DataFrame(titles, columns=['titles'])
    df_section_titles['category'] = category[i]
    df_fitles = pd.concat([df_fitles, df_section_titles], axis='rows',
                          ignore_index=True)
print(df_fitles.head())
df_fitles.info()
print(df_fitles['category'].value_counts())
df_fitles.to_csv('./crawling_data/naver_headline_news_{}.csv'.format(
    datetime.datetime.now().strftime('%Y%m%d')), index= False)



















