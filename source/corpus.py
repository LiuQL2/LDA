# /usr/bin/env python
# -*- coding:utf-8 -*-
"""
This is the file that defines Corpus class for LDA model.
"""

import sys
import os
import time
reload(sys)
sys.setdefaultencoding('utf-8')


class Corpus(object):
    """
    The class which is used to process corpus for LDA model.
    """
    def __init__(self):
        """
        Initialize an instance.
        """
        # documents in the corpus, key: id of document, value: list of id of all words in document..
        self.documents = {}
        # name of documents. key:id of document, value:name of document.
        self.documents_name_dict = {}
        # vocabulary. key:word, value:word id.
        self.word_id = {}
        # vocabulary. key:word id, value:word.
        self.id_word = {}

    def load_directory_corpus(self, directory,sep=' ',key_word_list=None, no_key_word_list=None):
        """
        The method to load corpus from directory, which contains the documents. And each document in the directory is an
        dependent file. Also, the document must has been tokenize and stop words were moved. And the character between
        two words is the parameter: sep.
        :param directory: the name of directory, which contains documents.
        :param sep: the character between two words in document. Default value is blank space.
        :param key_word_list: the string the must appeared in the name of document. Default value is None.
        :param no_key_word_list: the string the must not appeared in the name of document. Default value is None.
        :return: nothing.
        """
        document_name_list = self.get_dirlist(path=directory,key_word_list=key_word_list,
                                              no_key_word_list=no_key_word_list)
        for index in range(0, len(document_name_list),1):
            document_name = document_name_list[index]
            document = self.read_document(directory=directory, document_name=document_name,sep=sep)
            print 'loading document:', index, 'length:', len(document)
            self.documents[index] = document
            self.documents_name_dict[index] = document_name
        print 'number of documents:', len(self.documents)
        print 'number of terms:', len(self.word_id)

    def load_file_corpus(self,corpus_file,sep=' '):
        """
        The method that load corpus from one file, which contains all documents and whose one line corresponds to one
        document. Also, document must be tokenize and stop words are moved. And the character between two words is the
        parameter: sep.
        :param corpus_file: The file that contains all documents. It is the specific location where the file is.
        :param sep: the character between two words in document. Default value is blank space.
        :return: Nothing
        """
        file = open(corpus_file,mode='r')
        index = 0
        for line in file:
            word_list = line.replace('\n','').split(sep)
            document = []
            for word in word_list:
                if word not in self.word_id.keys():
                    word_id = len(self.word_id)
                    self.word_id[word] = word_id
                    self.id_word[word_id] = word
                else:
                    word_id = self.word_id[word]
                document.append(word_id)
            self.documents[index] = document
            self.documents_name_dict[index]=index
            index = index + 1
        file.close()

    def read_document(self,directory,document_name,sep=' '):
        """
        The method that read one document in the directory of corpus. It is called by the function:load_directory_corpus.
        :param directory: name of directory where the document is.
        :param document_name: name of document that need to be read.
        :param sep: character between two words in the document.
        :return: list of word id in the document.
        """
        document = []
        file = open(directory + document_name,mode='r')
        for line in file:
            words_list  = line.replace('\n','').split(sep)
            for word in words_list:
                if len(word) >=2:
                    if word not in self.word_id.keys():
                        word_id = len(self.word_id)
                        self.word_id[word] = word_id
                        self.id_word[word_id] = word
                    else:
                        word_id = self.word_id[word]
                    document.append(word_id)
                else:
                    pass
        file.close()
        return document

    def word_to_id(self, word):
        """
        The method translates an word to it's id.
        :param word: word needed to be translated.
        :return: id of the word.
        """
        return self.word_id[word]

    def id_to_word(self,term_id):
        """
        The method translates an id of word to word.
        :param term_id: id needed to be translated.
        :return: word of the id.
        """
        return self.id_word[term_id]

    @staticmethod
    def get_dirlist(path, key_word_list=None, no_key_word_list=None):
        """
        The method that gets name of all documents in an directory.
        :param path: The directory that needed to be read.
        :param key_word_list:the string the must appeared in the name of document. Default value is None.
        :param no_key_word_list:the string the must not appeared in the name of document. Default value is None.
        :return: list of names of all documents.
        """
        file_name_list = os.listdir(path)  # 获得原始json文件所在目录里面的所有文件名称
        if key_word_list == None and no_key_word_list == None:
            temp_file_list = file_name_list
        elif key_word_list != None and no_key_word_list == None:
            temp_file_list = []
            for file_name in file_name_list:
                have_key_words = True
                for key_word in key_word_list:
                    if key_word not in file_name:
                        have_key_words = False
                        break
                    else:
                        pass
                if have_key_words == True:
                    temp_file_list.append(file_name)
        elif key_word_list == None and no_key_word_list != None:
            temp_file_list = []
            for file_name in file_name_list:
                have_no_key_word = False
                for no_key_word in no_key_word_list:
                    if no_key_word in file_name:
                        have_no_key_word = True
                        break
                if have_no_key_word == False:
                    temp_file_list.append(file_name)
        elif key_word_list != None and no_key_word_list != None:
            temp_file_list = []
            for file_name in file_name_list:
                have_key_words = True
                for key_word in key_word_list:
                    if key_word not in file_name:
                        have_key_words = False
                        break
                    else:
                        pass
                have_no_key_word = False
                for no_key_word in no_key_word_list:
                    if no_key_word in file_name:
                        have_no_key_word = True
                        break
                    else:
                        pass
                if have_key_words == True and have_no_key_word == False:
                    temp_file_list.append(file_name)
        return temp_file_list