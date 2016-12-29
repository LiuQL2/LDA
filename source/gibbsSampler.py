# /usr/bin/env python
# -*- coding:utf-8 -*-
"""
The file that defines the class: GibbsSampler for LDA model.
"""
import sys
import random
import datetime
import pandas as pd
reload(sys)
sys.setdefaultencoding('utf-8')


class GibbsSampler(object):
    """
    The class, which is used to inference for LDA model by Gibbs Sampling.
    """
    def __init__(self,corpus,topic_number=10,iteration_number=1000,burn_in=500,update_cycle=100,alpha=None,beta=None):
        """
        The method that initializes an instance of the class.
        :param corpus: an instance of class: Corpus.
        :param topic_number: number of topic in LDA model.
        :param iteration_number: number of iteration when using Gibbs Sampling.
        :param burn_in: number of "burn in" iteration when using Gibbs Sampling.
        :param update_cycle: how often does we updates parameters of LDA after burn in.
        :param alpha: an vector, parameter of LDA, which determines the distribution of documents over topics in the
                       first and whose length is equal to the number of topics. Default value is None
        :param beta: an vector, parameter of LDA, which determines the distribution of topics over terms in the first
                      and whose length is equal to the number of terms. Default value is None.
        """
        # documents, key: id of document, value: list of word in an specific document.
        self.documents = corpus.documents
        # number of iteration when using Gibbs Sampling.
        self.iteration_number = iteration_number
        self.topic_number = topic_number
        self.burn_in = burn_in
        self.update_cycle = update_cycle
        # number of terms.
        self.term_number = len(corpus.word_id)
        # number of documents.
        self.document_number = len(self.documents)
        # if alpha and beta is None, then assign values to them.
        if alpha == None:
            self.alpha = [2.0] * self.topic_number
        else:
            self.alpha = alpha
        if beta == None:
            self.beta = [0.5] * self.term_number
        else:
            self.beta = beta
        # The sum of elements in beta.
        self.sum_beta = sum(self.beta)
        # The sum of elements in alpha.
        self.sum_alpha = sum(self.alpha)
        # counter, [m][k] refers to the number of times that topic k has been observed with a word in document m.
        self.document_topic_count_matrix = {}
        # counter, [k][t] refers to the number of times that term t has been observed with topic k.
        self.topic_term_count_matrix = {}
        # distribution matrix, [m][k] refers the probability that assigning topic k to document m.
        self.document_distribution_over_topic = {}
        # distribution matrix, [k][t] refers the probability that assigning topic k to term t.
        self.topic_distribution_over_term = {}
        # counter, [m] refers the number of times that all topics have been observed with a word in document m.
        # also, [m] equals to the number of words in document m.
        self.sum_document_by_topic_count = {}
        # counter, [k] refers the number of times that all terms have been observed with topic k.
        self.sum_topic_by_term_count = {}
        # topic assigned to an word in a document. [m][n] refers to the topic that assigned to the n th word in document
        # m.
        self.word_topic_assignment = {}
        # the number of times that the distribution has been updated.
        self.update_number = 0.0

    def gibbs_sample(self):
        """
        The method that realizes the Gibbs Sampling.
        :return: Nothing.
        """
        # Initialize the initial state of Markov Chain.
        self.initialize()
        # Gibbs Sampling.
        for iteration_index in range(0, self.iteration_number, 1):
            for m in range(0,self.document_number,1):
                for n in range(0, len(self.documents[m]), 1):
                    # Change the state of word_m_n according to it's full conditional probability.
                    self.sample_by_full_condition(m=m,n=n)
            print 'iteration:', iteration_index,datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            if iteration_index > self.burn_in and iteration_index % self.update_cycle == 0:
                # Update the distribution after burn in.
                self.update_distribution()
            else:
                pass
        # calculate the final distribution.
        self.get_distribution()

    def sample_by_full_condition(self, m, n):
        """
        The method that changes topic assigned to the n th word of document m.
        :param m: document m.
        :param n: sequence of word in document m.
        :return: Nothing.
        """
        # topic assigned to this word before.
        topic = self.word_topic_assignment[m][n]
        # each counter of the four counters minus one respectively.
        self.document_topic_count_matrix[m][topic] -= 1.0
        self.topic_term_count_matrix[topic][self.documents[m][n]] -= 1.0
        self.sum_document_by_topic_count[m] -= 1.0
        self.sum_topic_by_term_count[topic] -= 1.0

        # full conditional distribution of this word.
        full_condition_probability = []
        for k in range(0, self.topic_number, 1):
            phi = (self.topic_term_count_matrix[k][self.documents[m][n]] + self.beta[self.documents[m][n]]) /\
                  (self.sum_topic_by_term_count[k] + self.sum_beta)
            theta = (self.document_topic_count_matrix[m][k] + self.alpha[k]) /\
                    (self.sum_document_by_topic_count[m] + self.sum_alpha)
            probability = phi * theta
            full_condition_probability.append(probability)
        # cumulating the full conditional distribution.
        for index in range(1, len(full_condition_probability), 1):
            full_condition_probability[index] += full_condition_probability[index-1]
        # get new topic according to the full conditional distribution.
        random_double = random.uniform(0,1) * full_condition_probability[-1]
        for index in range(0, len(full_condition_probability),1):
            if random_double < full_condition_probability[index]:
                topic = index
                break
            else:
                pass
        # each counter of the four counters adds one respectively.
        self.document_topic_count_matrix[m][topic] += 1.0
        self.topic_term_count_matrix[topic][self.documents[m][n]] += 1.0
        self.sum_document_by_topic_count[m] += 1.0
        self.sum_topic_by_term_count[topic] += 1.0
        self.word_topic_assignment[m][n] = topic

    def initialize(self):
        """
        The method that initializes the initial state of Markov Chain. Also initializing the initial counters and topics
        assigned to all words of all documents.
        :return: Nothing.
        """
        # Initializing the counter and distribution.
        for k in range(0, self.topic_number,1):
            self.topic_term_count_matrix[k]= [0.0] * self.term_number
            self.topic_distribution_over_term[k] = [0.0] * self.term_number
            self.sum_topic_by_term_count[k] = 0.0
        for m in range(0, self.document_number,1):
            self.document_topic_count_matrix[m] = [0.0] * self.topic_number
            self.document_distribution_over_topic[m] = [0.0] * self.topic_number
            self.sum_document_by_topic_count[m] = 0.0

        # Initializing topics assigned to all words of all documents.
        for m in range(0, self.document_number, 1):
            N = len(self.documents[m])
            self.word_topic_assignment[m] = [-1] * N
            for n in range(0, N,1):
                topic = int(random.uniform(0,1) * self.topic_number)
                self.document_topic_count_matrix[m][topic] += 1.0
                self.topic_term_count_matrix[topic][self.documents[m][n]] += 1.0
                self.sum_topic_by_term_count[topic] += 1.0
                self.word_topic_assignment[m][n] = topic
            self.sum_document_by_topic_count[m] = N

    def update_distribution(self):
        """
        The method that updates distributions according to the four counters.
        :return: Nothing.
        """
        for m in range(0, self.document_number, 1):
            for k in range(0, self.topic_number, 1):
                self.document_distribution_over_topic[m][k]+=((self.document_topic_count_matrix[m][k] + self.alpha[k])
                                                              / (self.sum_document_by_topic_count[m] + self.sum_alpha))
        for k in range(0, self.topic_number, 1):
            for v in range(0, self.term_number,1):
                self.topic_distribution_over_term[k][v] += ((self.topic_term_count_matrix[k][v] + self.beta[v]) / \
                                                            (self.sum_topic_by_term_count[k] + self.sum_beta))
        # the number of times that the distributions are updated.
        self.update_number += 1

    def get_distribution(self):
        """
        The method that calculates final distribution.
        :return: Nonthing.
        """

        # If the distributions have been updated before.
        if self.update_number > 0:
            for m in range(0, self.document_number, 1):
                for k in range(0, self.topic_number,1):
                    probability = self.document_distribution_over_topic[m][k] / self.update_number
                    self.document_distribution_over_topic[m][k] = probability
            for k in range(0, self.topic_number,1):
                for v in range(0, self.term_number,1):
                    probability = self.topic_distribution_over_term[k][v] / self.update_number
                    self.topic_distribution_over_term[k][v] = probability
        # The distributions have not been updated once.
        else:
            for m in range(0, self.document_number, 1):
                for k in range(0, self.topic_number, 1):
                    self.document_distribution_over_topic[m][k] = (
                    (self.document_topic_count_matrix[m][k] + self.alpha[k]) / (
                    self.sum_document_by_topic_count[m] + self.sum_alpha))
            for k in range(0, self.topic_number, 1):
                for v in range(0, self.term_number, 1):
                    self.topic_distribution_over_term[k][v] = (
                    (self.topic_term_count_matrix[k][v] + self.beta[v]) / (
                    self.sum_topic_by_term_count[k] + self.sum_beta))

    def get_topic_distribution_over_term(self):
        """
        The method that gets the distribution of topics over terms.
        :return: The distribution of topics over terms.
        """
        return self.topic_distribution_over_term

    def get_document_distribution_over_topic(self):
        """
        The method that gets the distribution of documents over topics.
        :return: The distribution of documents over topics.
        """
        return self.document_distribution_over_topic
