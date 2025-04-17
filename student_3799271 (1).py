############ CODE BLOCK 0 ################
# ^ DO NOT CHANGE THIS LINE

# You are not allowed to add additional imports!
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn import datasets, ensemble, metrics, svm, model_selection, linear_model

############ CODE BLOCK 1 ################
# ^ DO NOT CHANGE THIS LINE

def training_test_split(X, y, test_size=0.3, random_state=None):
    """ Split the features X and labels y into training and test features and labels. 
    
    `split` indicates the fraction (rounded down) that should go to the test set.

    `random_state` allows to set a random seed to make the split reproducible. 
    If `random_state` is None, then no random seed will be set.
    
    """
    X, y = np.array(X), np.array(y)  # evern though the X and y are defined as np now, we still do this for safety
    if random_state:
        np.random.seed(random_state)

    sample_n = len(X)
    test_n = int(sample_n * test_size)

    sample_indices = np.arange(sample_n)
    np.random.shuffle(sample_indices)

    test_indices = sample_indices[:test_n]
    train_indices = sample_indices[test_n:]

    X_train = X[train_indices]
    X_test = X[test_indices]
    
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    return X_train, X_test, y_train, y_test

############ CODE BLOCK 2 ################
# ^ DO NOT CHANGE THIS LINE

def true_positives(true_labels, predicted_labels, positive_class):
    pos_true = true_labels == positive_class  # compare each true label with the positive class
    pos_predicted = predicted_labels == positive_class # compare each predicted label to the positive class
    match = pos_true & pos_predicted # use logical AND (that's the `&`) to find elements that are True in both arrays
    return np.sum(match)  # count them

############ CODE BLOCK 3 ################
# ^ DO NOT CHANGE THIS LINE

def false_positives(true_labels, predicted_labels, positive_class):
    pos_predicted = predicted_labels == positive_class  # predicted to be positive class
    neg_true = true_labels != positive_class  # actually negative class
    match = pos_predicted & neg_true  # The `&` is element-wise logical AND
    return np.sum(match)  # count the number of matches

############ CODE BLOCK 4 ################
# ^ DO NOT CHANGE THIS LINE

def true_negatives(true_labels, predicted_labels, negative_class):  
    # raise NotImplementedError('Your code here')
    # samples that are actually negative and predicted as negative
    actual_neg = true_labels != positive_class  #  actually negative
    predicted_neg = predicted_labels != positive_class  # predicted as negative
    match = actual_neg & predicted_neg  #  true negatives
    return np.sum(match)  # number of true negatives

    
def false_negatives(true_labels, predicted_labels, negative_class):
# False negatives: samples that are actually positive but predicted as negative
    actual_pos = true_labels == positive_class  # samples that are actually positive
    predicted_neg = predicted_labels != positive_class  # samples predicted as negative
    match = actual_pos & predicted_neg  # logical AND to find false negatives
    return np.sum(match)  # count the number of false negatives

############ CODE BLOCK 5 ################
# ^ DO NOT CHANGE THIS LINE

def precision(true_labels, predicted_labels, positive_class):
    TP = true_positives(true_labels, predicted_labels, positive_class)
    FP = false_positives(true_labels, predicted_labels, positive_class)
    return TP / (TP + FP)

############ CODE BLOCK 6 ################
# ^ DO NOT CHANGE THIS LINE

def recall(true_labels, predicted_labels, positive_class):
    # raise NotImplementedError('Your code here')
    TP = true_positives(true_labels, predicted_labels, positive_class)
    FN = false_negatives(true_labels, predicted_labels, positive_class)
    if TP + FN == 0 :
        return 0
    return TP/(TP + FN)

############ CODE BLOCK 7 ################
# ^ DO NOT CHANGE THIS LINE

def accuracy(true_labels, predicted_labels, positive_class):
    # raise NotImplementedError('Your code here')
    TP = true_positives(true_labels, predicted_labels, positive_class)
    FP = false_positives(true_labels, predicted_labels, positive_class)
    TN = true_negatives(true_labels, predicted_labels, positive_class)
    FN = false_negatives(true_labels, predicted_labels, positive_class)

    total_samples = TP + TN + FP + FN

    if total_samples == 0:
        return 0
        
    return (TP + TN) / total_samples

############ CODE BLOCK 8 ################
# ^ DO NOT CHANGE THIS LINE

def specificity(true_labels, predicted_labels, positive_class):
    # raise NotImplementedError('Your code here')
    # TP = true_positives(true_labels, predicted_labels, positive_class)
    FP = false_positives(true_labels, predicted_labels, positive_class)
    TN = true_negatives(true_labels, predicted_labels, positive_class)
    # FN = false_negatives(true_labels, predicted_labels, positive_class)

    return TN/(TN + FP)

############ CODE BLOCK 9 ################
# ^ DO NOT CHANGE THIS LINE

def balanced_accuracy(true_labels, predicted_labels, positive_class):
    # raise NotImplementedError('Your code here')
    TP = true_positives(true_labels, predicted_labels, positive_class)
    FP = false_positives(true_labels, predicted_labels, positive_class) 
    TN = true_negatives(true_labels, predicted_labels, positive_class)
    FN = false_negatives(true_labels, predicted_labels, positive_class)

    TPR = TP/ (TP + FN)
    TNR = TN/ (TN + FP)
    print('TPR:', TPR)
    print('TNR:', TNR)
    return (TPR + TNR)/2

print(balanced_accuracy(y_test, y_pred, 'A'))

############ CODE BLOCK 10 ################
# ^ DO NOT CHANGE THIS LINE

def F1(true_labels, predicted_labels, positive_class):
    # raise NotImplementedError('Your code here
    pre = precision(true_labels, predicted_labels, positive_class)
    rec = recall(true_labels, predicted_labels, positive_class)
    return 2 * (pre * rec)/ (pre + rec)

