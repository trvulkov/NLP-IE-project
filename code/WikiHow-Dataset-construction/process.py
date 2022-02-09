import pandas as pd
import os
import re

# read data from the csv file (from the location it is stored)
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
ARTICLES_DIR = os.path.join(CURRENT_DIR, 'articles')
SUMMARIES_DIR = os.path.join(ARTICLES_DIR, 'summaries')

csv_path = os.path.join(CURRENT_DIR, 'wikihowAll.csv')

df = pd.read_csv(csv_path)
df = df.astype(str)

if not os.path.exists(ARTICLES_DIR):
    os.makedirs(ARTICLES_DIR)
if not os.path.exists(SUMMARIES_DIR):
    os.makedirs(SUMMARIES_DIR)

# go over the all the articles in the data file, but only pick a subset since there's way too many

counter = 0
for index, row in df.iterrows():
    counter += 1
    if counter % 500 != 0:
        continue    

    filename = row['title']
    summary = row['headline'] # headline is the column representing the summary sentences
    article = row['text'] # text is the column representing the article


    #  a threshold is used to remove short articles with long summaries as well as articles with no summary
    if len(summary) >= (0.75 * len(article)):
        continue

    # remove extra commas
    summary = summary.replace('.,', '.')
    #summary = re.sub('\s+',' ', summary)
    summary = re.sub('\n+','\n', summary)
    summary = summary.strip()
    article = re.sub(r'[.]+[\n]+[,]', '.\n', article)
    #article = re.sub('\s+',' ', article)
    article = re.sub('\n+','\n', article)
    article = article.strip()

    try:
        summary = summary.encode('ascii', 'replace')
    except:
        continue
    try:
        article = article.encode('ascii', 'replace')
    except:
        continue

    filename = ''.join(x for x in filename if x.isalnum()) + '.txt'
    try:
        print(filename)
    except:
        continue

    with open(os.path.join(ARTICLES_DIR, filename), 'wb+') as article_file, \
         open(os.path.join(SUMMARIES_DIR, filename), 'wb+') as summary_file:
        article_file.write(article)
        summary_file.write(summary)