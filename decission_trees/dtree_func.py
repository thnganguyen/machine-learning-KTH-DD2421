import numpy as np
import pandas as pd
import random
import monkdata as m
import dtree as d
from drawtree_qt5 import drawTree

def compute_entropy(datasets_names, datasets):
    for dataset_name, dataset in zip(datasets_names, datasets):
        print(dataset_name, ':' , round(d.entropy(dataset),3))


def entropy_matrix(datasets, attribute_index, max_att_list):
    entropy_matrix = np.zeros((len(datasets), len(m.attributes[attribute_index].values)))
    for idx, dataset in enumerate(datasets):
        att = m.attributes[max_att_list[idx]]
        for j, v in enumerate(att.values):
            entropy_matrix[idx,j] = d.entropy(d.select(dataset, att, v))
    print(entropy_matrix)


def information_gain(datasets):
    attributes_names = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6']
    information_gain_matrix = np.zeros((len(datasets), len(m.attributes)))
    for idx, dataset in enumerate(datasets):
        for i in range(len(attributes_names)):
            average_gain = round(d.averageGain(dataset, m.attributes[i]),4)
            information_gain_matrix[idx, i] = average_gain
    return information_gain_matrix


def maximum_information_gain(datasets):
    attributes_names = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6']
    information_gain_matrix = information_gain(datasets)
    maximum_information_gain_matrix = np.zeros((len(datasets),2),dtype=object)
    max_att_list = []
    for i in range(len(datasets)):
        inf_gain_maximum = max(information_gain_matrix[i])
        if inf_gain_maximum == 0:
            e = 0,0
            max_att_list.append(0)
        if inf_gain_maximum != 0:
            x, = np.where(information_gain_matrix[i] == inf_gain_maximum)
            e = attributes_names[int(x)], inf_gain_maximum
            maximum_information_gain_matrix[i] = e
            max_att_list.append(int(x))
    return maximum_information_gain_matrix, max_att_list


def split_tree_by_attribute_and_value(dataset, attribute_idx):
    attribute_values = m.attributes[attribute_idx].values
    attribute_values_list = [[i] for i in list(attribute_values)]
    dataset_by_attribute_and_value = []
    for value in attribute_values:
        dataset_by_attribute_and_value.append(d.select(dataset, m.attributes[attribute_idx], value))
    return dataset_by_attribute_and_value, attribute_values_list


def perform_buildTree(datasets):
    datasets_trees = []
    for dataset in datasets:
        datasets_trees.append(d.buildTree(dataset, m.attributes))
    return datasets_trees


def check_correct_incorrect_classification(datasets, test_datasets, datasets_names):
    datasets_trees = perform_buildTree(datasets)
    check = {}
    check_e = np.zeros((len(datasets), 2))
    for i, dataset, dataset_name, dataset_tree, test_dataset in zip(range(len(datasets)),datasets, datasets_names,
                                                                    datasets_trees, test_datasets):
        correct_classification = round(d.check(dataset_tree, test_dataset),3)
        check[dataset_name] = correct_classification
        err = round(1 - d.check(dataset_tree, dataset),3), round((1 - correct_classification),3)
        check_e[i] = err
    return check, check_e


def partition(data, fraction):
    ldata = list(data)
    random.shuffle(ldata)
    breakPoint = int(len(ldata) * fraction)
    return ldata[:breakPoint], ldata[breakPoint:]
