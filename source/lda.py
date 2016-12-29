# /usr/bin/env python
# -*- coding:utf-8 -*-

"""
The file that defines the class: LDA.
"""
import sys
import os
import csv
from corpus import Corpus
from gibbsSampler import GibbsSampler
reload(sys)
sys.setdefaultencoding('utf-8')


class LDA(object):
    """
    The class: LDA, a topic model.
    """
    def __init__(self):
        """
        Initializing an instance of LDA.
        """
        # corpus of a LDA model.
        self.corpus = Corpus()
        # the distribution of topics over terms.
        self.topic_distribution_over_term = None
        # the distribution of documents over topics.
        self.document_distribution_over_topic = None
        # gibbs sampler.
        self.gibbs_sampler = None

    def load_directory_corpus(self,directory,key_word_list=None,no_key_word_list=None,sep=' '):
        """
        The method to load corpus from directory, which contains the documents. And each document in the directory is an
        dependent file. Also, the document must has been tokenize and stop words were moved. And the character between
        two words is the parameter: sep.
        This method calls an function of corpus.
        :param directory: the name of directory, which contains documents.
        :param key_word_list: the string the must appeared in the name of document. Default value is None.
        :param no_key_word_list: the string the must not appeared in the name of document. Default value is None.
        :param sep: the character between two words in document. Default value is blank space.
        :return: nothing.
        """
        self.corpus.load_directory_corpus(directory=directory, key_word_list=key_word_list,
                                          no_key_word_list=no_key_word_list,sep=sep)

    def load_file_corpus(self,corpus_file,sep=' '):
        """
        The method that load corpus from one file, which contains all documents and whose one line corresponds to one
        document. Also, document must be tokenize and stop words are moved. And the character between two words is the
        parameter: sep.
        This method calls an function of corpus
        :param corpus_file:The file that contains all documents. It is the specific location where the file is.
        :param sep:the character between two words in document. Default value is blank space.
        :return:Nothing
        """
        self.corpus.load_file_corpus(corpus_file=corpus_file,sep=sep)

    def train_model(self,topic_number=10,iteration_number=1000,burn_in=500,update_cycle=100):
        """
        The method that trains LDA model.
        :param topic_number: number of topics.
        :param iteration_number: number of iterations.
        :param burn_in: number of iterations that belong to the phrase of burn in.
        :param update_cycle: how often does we updates parameters of LDA after burn in.
        :return: Nothing.
        """
        alpha = [2.0] * topic_number
        beta = [0.1] * len(self.corpus.id_word)
        # Initializing the gibbs sampler.
        self.gibbs_sampler = GibbsSampler(corpus=self.corpus,topic_number=topic_number,iteration_number=iteration_number,
                                           burn_in=burn_in,update_cycle=update_cycle,alpha=alpha,beta=beta)
        # Gibbs Sampling.
        self.gibbs_sampler.gibbs_sample()
        # Get the distribution of topics over terms.
        self.topic_distribution_over_term = self.gibbs_sampler.get_topic_distribution_over_term()
        # Get the distribution of documents over topics.
        self.document_distribution_over_topic = self.gibbs_sampler.get_document_distribution_over_topic()

    def output_topic(self, term_number=10,save_topic=False,save_file_name='topic_distribution_over_terms.txt'):
        """
        The method that prints or saves the distribution of topics over terms.
        :param term_number: number of terms printed for each topic. Default value is 10.
        :param save_topic: boolean value, whether to save the distribution to an file.
        :param save_file_name: the name of file that contains the distribution if save_topic is True.
        :return: Nothing.
        """
        # Printing the distribution.
        for topic in range(0, len(self.topic_distribution_over_term), 1):
            probability_list = self.topic_distribution_over_term[topic]
            term_prob_dict = {}
            for index in range(0,self.gibbs_sampler.term_number,1):
                term_prob_dict[index] = probability_list[index]
            probability_list = sorted(term_prob_dict.iteritems(), key=lambda d:d[1], reverse = True)[0:term_number]
            print '\n' + '#' * 50
            print 'topic:', topic
            for term,probability in probability_list:
                print self.corpus.id_to_word(term),':', probability
        # Saving the distribution to an file.
        if save_topic==True:
            path = os.getcwd().replace('\\','/').split('/LDA/')[0] + '/LDA/result/'
            file = open(path + save_file_name,'wb')
            for topic in range(0, len(self.topic_distribution_over_term), 1):
                probability_list = self.topic_distribution_over_term[topic]
                term_prob_dict = {}
                for index in range(0, self.gibbs_sampler.term_number, 1):
                    term_prob_dict[index] = probability_list[index]
                probability_list = sorted(term_prob_dict.iteritems(), key=lambda d: d[1], reverse=True)[0:term_number]
                file.write('#' * 50 + '\n')
                file.write('topic:'+ str(topic) + '\n')
                for term, probability in probability_list:
                    file.write(self.corpus.id_to_word(term) + ':' + str(probability) + '\n')
            file.close()
        else:
            pass

    def ouput_document(self,document_number=None,save_document=False,
                       save_file_name='document_distribution_over_topics.csv'):
        """
        The method that prints or saves the distribution of documents over topics.
        :param document_number: number of documents that needed to be printed or saved. Default value is None.
        :param save_document: boolean value, whether to save the distribution to an file.
        :param save_file_name: the name of file that contains the distribution if save_document is True.
        :return: Nothing.
        """
        # get the number of documents that needed to be printed or saved when the default value of document_number is
        # None.
        if document_number == None:
            document_number = len(self.corpus.documents)
        else:
            pass
        # Printing the distribution.
        for document in range(0, self.gibbs_sampler.document_number, 1)[0:document_number]:
            print '\n' + '#' * 50
            print 'document:',self.corpus.documents_name_dict[document]
            print self.document_distribution_over_topic[document]
        # Saving the distribution to an file.
        if save_document==True:
            path = os.getcwd().replace('\\','/').split('/LDA/')[0] + '/LDA/result/'
            file = open(path + save_file_name,'wb')
            writer = csv.writer(file)
            head = ['document']
            for topic in range(0,self.gibbs_sampler.topic_number,1):
                head.append('topic_' + str(topic))
            writer.writerow(head)
            for document in range(0, self.gibbs_sampler.document_number, 1):
                row = [self.corpus.documents_name_dict[document]]
                row = row +  self.document_distribution_over_topic[document]
                writer.writerow(row)
            file.close()
        else:
            pass

if __name__ == '__main__':
    print os.getcwd().replace('\\','/').replace('source','result/')