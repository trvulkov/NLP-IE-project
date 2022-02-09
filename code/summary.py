import os
import argparse
from pprint import pprint
from abc import ABC

import spacy
import pytextrank
from rouge import Rouge

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer as Summarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words


SENTENCES_IN_SUMMARY = 3

class Corpus:
    def __init__(self, corpus_name, start = -1, end = -1):
        self.project_dir = os.path.dirname(os.path.abspath(__file__))

        self.corpora_dir = os.path.join(self.project_dir, 'corpus')
        self.corpus_dir = os.path.join(self.corpora_dir, corpus_name)

        self.summary_dir = os.path.join(self.corpus_dir, 'summaries')

        self.titles = []
        self.texts = []
        self.reference_summaries = []
        self.doc_count = 0

        self.read_corpus(start, end)

        self.rouge = Rouge() # initialize ROUGE

    def read_corpus(self, start, end):
        file_names = next(os.walk(self.corpus_dir))[2]

        if start == -1:
            start = 0
        if end == -1:
            end = len(file_names)

        for file_name in file_names[start:end]:
            with open(os.path.join(self.corpus_dir, file_name)) as text_file, \
                 open(os.path.join(self.summary_dir, file_name)) as summary_file:
                text = text_file.read().strip()
                summary = summary_file.read().strip()
                
            self.doc_count += 1
            self.titles.append(file_name)
            self.texts.append(text)
            self.reference_summaries.append(summary)

    def get_rouge(self, output_summaries): # output_summaries and reference_summaries need to have the same length        
        scores_individual = self.rouge.get_scores(output_summaries, self.reference_summaries)
        score_averaged = self.rouge.get_scores(output_summaries, self.reference_summaries, avg = True)

        return scores_individual, score_averaged


    def score(self, output_summaries):
        scores_individual, score_averaged = self.get_rouge(output_summaries)

        return self.texts, output_summaries, scores_individual, score_averaged
   
    def score_and_print(self, output_summaries):
        scores_individual, score_averaged = self.get_rouge(output_summaries)

        for text, summary, score in zip(self.texts, output_summaries, scores_individual):
            print(text)
            print('--------------------------------------------------------------')

            print(summary)
            pprint(score)
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')

        print('--------AVERAGE SCORE-----------------------------------------')
        pprint(score_averaged)

    def score_and_write(self, output_summaries, folder_name):
        scores_individual, score_averaged = self.get_rouge(output_summaries)

        path = os.path.join(self.corpus_dir, 'output for - ' + folder_name + ', with ' + str(SENTENCES_IN_SUMMARY) + ' sentence limit')
        if not os.path.exists(path):
            os.makedirs(path)

        for title, text, summary, score in zip(self.titles, self.texts, output_summaries, scores_individual):
            with open(os.path.join(path, title), 'w+') as file:
                file.write(text)
                file.write('\n--------------------------------------------------------------\n')

                file.write(summary)
                file.write('\n--------------------------------------------------------------\n')
                
                pprint(score, stream = file)
                file.write('\n--------AVERAGE SCORE-----------------------------------------\n')
                pprint(score_averaged, stream = file)    


class AbstractSummarizer(ABC):
    def process_text(self, text):
        raise NotImplementedError

    def process_corpus(self, corpus, write = False):
        raise NotImplementedError

class TextrankSummarizer(AbstractSummarizer):
    def __init__(self, summarizer_type, model):
        self.summarizer_type = summarizer_type
        self.model = model

        self.nlp = spacy.load(model) # load a spaCy model for English
        self.nlp.add_pipe(summarizer_type) # add PyTextRank to the spaCy pipeline - can choose textrank, positionrank or biasedtextrank

    def check_pipes(self):
        print(self.nlp.pipe_names)

    def get_summary(self, doc): # doc is a spaCy Doc object, returned by self.nlp after processing some text
        summary = doc._.textrank.summary(limit_phrases = 15, limit_sentences = SENTENCES_IN_SUMMARY)

        result = ''
        for sentence in summary:
            result += sentence.text

        return result        

    def process_text(self, text):
        doc = self.nlp(text)
        summary = self.get_summary(doc)

        return summary

    def process_corpus(self, corpus, write = False):
        docs = list(self.nlp.pipe(corpus.texts)) # more efficient for multiple docs
        output_summaries = [self.get_summary(doc) for doc in docs]

        if write:
            corpus.score_and_write(output_summaries, self.summarizer_type + '_' + self.model)
        else:
            corpus.score_and_print(output_summaries)

class LSASummarizer(AbstractSummarizer):
    def __init__(self, language):
        self.language = language

        self.stemmer = Stemmer(self.language)
        self.summarizer = Summarizer(self.stemmer)
        self.summarizer.stop_words = get_stop_words(language)

    def process_text(self, text):
        doc = PlaintextParser.from_string(text, Tokenizer(self.language))

        summary_sentences = self.summarizer(doc.document, SENTENCES_IN_SUMMARY)
        summary = ''.join([str(sent) + '\n' for sent in summary_sentences])

        return summary

    def process_corpus(self, corpus, write = False):
        docs = [PlaintextParser.from_string(text, Tokenizer(self.language)) for text in corpus.texts]
        output_summaries_as_sentences = [self.summarizer(doc.document, SENTENCES_IN_SUMMARY) for doc in docs]

        output_summaries = [''.join([str(sent) + '\n' for sent in summary]) for summary in output_summaries_as_sentences]

        if write:
            corpus.score_and_write(output_summaries, 'lsa')
        else:
            corpus.score_and_print(output_summaries)


ap = argparse.ArgumentParser()
ap.add_argument('-d','--dataset', type = str, default = 'DUC', choices = ['DUC', 'WikiHow', 'MultiNews'], 
                help = 'Name of the dataset to be used - one of the corpora in the corpus folder')
ap.add_argument('-r', '--range', type = int, nargs = 2, default = [-1, -1], 
                help = '''Two numbers, a < b, forming an interval (of document numbers, indexed from 0), in order to only process a subset of the corpus.
                If a is -1, the interval is [0,b), and if b is -1, the interval is [a, number of documents in the corpus)''')                
ap.add_argument('-t', '--type', type = str, default = 'textrank', choices = ['textrank', 'positionrank', 'biasedtextrank', 'lsa'], 
                help = 'The type of summarization to use - TextRank, PositionRank, BiasedTextRank, or LSA')
ap.add_argument('-m', '--model', type = str, default = 'en_core_web_sm', choices = ['en_core_web_sm', 'en_core_web_md', 'en_core_web_lg'], 
                help = 'In the case of one of the *Rank algorithms - the name of the spaCy model to be used (en_core_web_sm/md/lg, 3 different sizes)')
ap.add_argument('-s', '--sentences', type = int, default = 3,
                help = 'The size (in amount of sentences) of the desired summary')
ap.add_argument('-w', '--write', action = 'store_true', default = False,
                help = 'Write the results to files (in the corresponding corpus directory) instead of printing them to the console')

args = vars(ap.parse_args())

SENTENCES_IN_SUMMARY = args['sentences']
start = args['range'][0]
end = args['range'][1]

corpus = Corpus(args['dataset'], start, end)

if args['type'] == 'lsa':
    summarizer = LSASummarizer('english')
else:
    summarizer = TextrankSummarizer(args['type'], args['model'])

summarizer.process_corpus(corpus, args['write'])

