# /usr/bin/env python
# -*- coding:utf-8 -*-
"""
This is used to test LDA, whose corpus is the directory where all documents are.
"""
import sys
from source.lda import LDA
reload(sys)
sys.setdefaultencoding('utf-8')


if __name__ == '__main__':
    directory = 'D:/Qianlong/PyCharmProjects/LDA/data/'
    lda = LDA()
    # Loading corpus from directory where all documents are.
    lda.load_directory_corpus(directory=directory,key_word_list=['txt'],no_key_word_list=None,sep=' ')
    lda.train_model(topic_number=10,iteration_number=1000,burn_in=500,update_cycle=50)
    lda.output_topic(term_number=10,save_topic=True)
    lda.ouput_document(document_number=None,save_document=True)
