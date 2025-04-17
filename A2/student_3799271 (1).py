############ CODE BLOCK 0 ################
#^ DO NOT CHANGE THIS LINE

import numpy as np
import pandas as pd
import pygraphviz as pgv
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, export_graphviz
from sklearn.metrics import accuracy_score

############ CODE BLOCK 1 ################
#^ DO NOT CHANGE THIS LINE

def gini(labels):
    # raise NotImplementedError('Your code here')

    if labels is None or len (labels) == 0: # if there is no labels or no instance of classes
        return 1.0
    
    # frequency = pd.Series(labels).value_counts().values  # counts the number of all classes
    frequency = labels.value_counts().values
    if len(frequency) == 1: # if there is only one class
        return 0.0

    probabilities = frequency / len(labels)
    sum_sq_prob = sum(prob ** 2 for prob in probabilities)
    impurity = 1 - sum_sq_prob
    return impurity

############ CODE BLOCK 2 ################
#^ DO NOT CHANGE THIS LINE

def entropy(labels):
    # raise NotImplementedError('Your code here')
    if labels is None or len (labels) == 0: # if there is no labels or no instance of classes
        return 0.0
    
    # frequency = pd.Series(labels).value_counts().values  # counts the number of all classes
    frequency = labels.value_counts().values
    if len(frequency) == 1: # if there is only one class
        return 0.0

    probabilities = frequency / len(labels)
    mul_log_p = [probabilities * np.log2(probabilities)] # use numpy to calculate log
    entropy = -1 * np.sum(mul_log_p)
    return entropy

############ CODE BLOCK 3 ################
#^ DO NOT CHANGE THIS LINE

class DTree:
    def __init__(self, metric):
        """Set up a new tree.
        
        We use the `metric` parameter to supply an impurity measure such as Gini or Entropy.
        The other class variables should be set by the "fit" method.
        """
        self._metric = metric  # what are we measuring impurity with? (Gini, Entropy, Minority Class...)
        self._samples = None  # how many training samples reached this node?
        self._distribution = []  # what was the class distribution in this node?
        self._label = None  # What was the majority class of training samples that reached this node?
        self._impurity = None  # what was the impurity at this node?
        self._split = False  # if False, then this is a leaf. If you branch from this node, use this to store the name of the feature you're splitting on.
        self._yes = None  # Holds the "yes" DTree object; None if this is still a leaf node
        self._no = None # Holds the "no" DTree object; None if this is still a leaf node
        

    def _best_split(self, features, labels):
        """ Determine the best feature to split on.

        :param features: a pd.DataFrame with named training feature columns
        :param labels: a pd.Series or pd.DataFrame with training labels
        :return: `best_so_far` is a string with the name of the best feature,
        and `best_so_far_impurity` is the impurity on that feature

        For each candidate feature the weighted impurity of the "yes" and "no"
        instances for that feature are computed using self._metric.

        We select the feature with the lowest weighted impurity.
        """
        # raise NotImplementedError('Your code here')

        #return values
        best_so_far = None
        best_so_far_impurity = float('inf')
        for feature in features.columns:
            yes_mask = features[feature] == 1
            no_mask = features[feature] == 0

            yes_labels = labels[yes_mask]
            no_labels = labels[no_mask]

            if not yes_labels.empty and not no_labels.empty:
                impurity_yes = self._metric(yes_labels)
                impurity_no = self._metric(no_labels)

                p_yes = len(yes_labels) / len(labels)
                p_no = len(no_labels) / len(labels)
                weighted_impurity = p_yes * impurity_yes + p_no * impurity_no

                if weighted_impurity < best_so_far_impurity:
                    best_so_far_impurity = weighted_impurity
                    best_so_far = feature
        return best_so_far, best_so_far_impurity

    def fit(self, features, labels):
        """ Generate a decision tree by recursively fitting & splitting them

        :param features: a pd.DataFrame with named training feature columns
        :param labels: a pd.Series or pd.DataFrame with training labels
        :return: Nothing.

        First this node is fitted as if it was a leaf node: the training majority label, number of samples,
        class distribution and impurity.

        Then we evaluate which feature might give the best split.

        If there is a best split that gives a lower weighed impurity of the child nodes than the impurity in this node,
        initialize the self._yes and self._no variables as new DTrees with the same metric.
        Then, split the training instance features & labels according to the best splitting feature found,
        and fit the Yes subtree with the instances that split to the True side,
        and the No subtree with the instances that are False according to the splitting feature.
        """
        
        # raise NotImplementedError('Your code here')

        # base case, reach the leaf
        if features.empty or labels.empty:
            return

        # 
        self._samples = len(labels)
        unique_labels = sorted(set(labels))
        label_counts = labels.value_counts().reindex(unique_labels, fill_value=0)
        self._distribution = label_counts.tolist()
        self._label = labels.mode()[0]
        self._impurity = self._metric(labels)

        # 
        split, split_loss = self._best_split(features, labels)
        if split is not None and split_loss < self._impurity:
            self._split = split

            mask_yes = features[split] == 1
            mask_no = features[split] == 0

            # 
            self._yes = DTree(self._metric)
            self._no = DTree(self._metric)
            self._yes.fit(features[mask_yes], labels[mask_yes])
            self._no.fit(features[mask_no], labels[mask_no])
        else:
            self._split = False
            self._yes = None
            self._no = None

    def predict(self, features):
        """ Predict the labels of the instances based on the features

        :param features: pd.DataFrame of test features
        :return: predicted labels

        We start by initializing an array of labels where we naively predict this node's label.
        The datatype of this array is set to `object` because otherwise numpy
        might select the minimum needed string length for the current label, regardless of child labels.

        Then if this is not a leaf node, we overwrite those values with the values of Yes and No child nodes,
        based on the feature split in this node.
        """
        results = np.full(features.shape[0], self._label, dtype=object)  # object!!!
        if self._split:  # branch node; recursively replace predictions with child predictions
            yes_index = features[self._split] > 0.5
            results[yes_index] = self._yes.predict(features.loc[yes_index])
            results[~yes_index] = self._no.predict(features.loc[~yes_index])
        return results

    def to_text(self, depth=0):
        if self._split:
            text = f'{"|   " * depth}|---{self._split} = no\n'
            text += self._no.to_text(depth=depth+1)
            text += f'{"|   " * depth}|---{self._split} = yes\n'
            text += self._yes.to_text(depth=depth+1)
            
        else:
            text = f'{"|   " * depth}|---{self._label} ({self._samples})\n'.upper()
        return text

    def to_graphviz(self, choice='', parent='R', graph=None, size='15,15'):
        details = f'\n\nimpurity = {self._impurity:.2f}\nsamples = {self._samples}\n{self._distribution}'
        if self._split:
            label = f'({self._label.lower()})'
        else:
            label = self._label.upper()
        if graph is None:  # root node
            graph = pgv.AGraph(directed=True)  # initialize the graph
            graph.graph_attr.update(size=size)
            graph.graph_attr.update(ratio='1.0')
        if self._split:  # branching nodes
            node_label = f'{label}\n{self._split.upper()}???{details}'  # display name
            graph.add_node(n=parent+choice, label=node_label, shape='diamond')
            self._yes.to_graphviz(choice='yes', parent=parent+choice, graph=graph)
            self._no.to_graphviz(choice='no', parent=parent+choice, graph=graph)
        else:  # leaf node
            node_label = f'{label}{details}'  # display name
            graph.node_attr.update(name=parent+choice, label=node_label, shape='rectangle')
        if choice != '':
            graph.add_edge(parent, parent + choice, label=choice)  # draw arrow from parent to this one
        return graph

