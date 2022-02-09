import os

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
ARTICLES_DIR = os.path.join(CURRENT_DIR, 'articles')
SUMMARIES_DIR = os.path.join(ARTICLES_DIR, 'summaries')

if not os.path.exists(ARTICLES_DIR):
    os.makedirs(ARTICLES_DIR)
if not os.path.exists(SUMMARIES_DIR):
    os.makedirs(SUMMARIES_DIR)

articles_path = os.path.join(CURRENT_DIR, 'test.txt.src.tokenized.fixed.cleaned.final.truncated.txt')
summaries_path = os.path.join(CURRENT_DIR, 'test.txt.tgt.tokenized.fixed.cleaned.final.truncated.txt')


counter = 0
with open(articles_path, encoding = 'ascii', errors = 'ignore') as articles_file:
    for line in articles_file:
        formatted_line = '.\n'.join(line.split('.'))

        with open(os.path.join(ARTICLES_DIR, str(counter) + '.txt') ,'w+') as current_article:
            current_article.write(formatted_line)

        counter += 1

counter = 0
with open(summaries_path, encoding = 'ascii', errors = 'ignore') as summaries_file:
    for line in summaries_file:
        formatted_line = '.\n'.join(line.split('.'))

        with open(os.path.join(SUMMARIES_DIR, str(counter) + '.txt') ,'w+') as current_article:
            current_article.write(formatted_line)

        counter += 1
