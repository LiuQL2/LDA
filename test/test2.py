# /usr/bin/env python
# -*- coding:utf-8 -*-
"""
This is used to test LDA model, whose corpus is an file whose one line corresponding to one document.
"""
import sys
from source.lda import LDA
reload(sys)
sys.setdefaultencoding('utf-8')


if __name__ == '__main__':
    file_corpus = 'D:/Qianlong/PyCharmProjects/LDA/data/file_corpus/corpus.txt'
    lda = LDA()
    # Loading corpus from an file.
    lda.load_file_corpus(corpus_file=file_corpus,sep=' ')
    lda.train_model(topic_number=10,iteration_number=100,burn_in=50,update_cycle=10)
    lda.output_topic(term_number=10,save_topic=True)
    lda.ouput_document(document_number=10,save_document=True)