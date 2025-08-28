# Data Handling & Visualization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from math import log2, sqrt

# Machine Learning (Scikit-learn)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics

# Visualization of Decision Tree
from six import StringIO
from IPython.display import Image
import pydotplus

# Progress Bar
from tqdm.notebook import tqdm

def entropy(class_y):
    """
    Input:
    - class_y: list of class labels (0's and 1's)

    Output:
    - The entropy

    Compute the entropy for a list of classes
    """
    if len(class_y) <= 1:
        return 0
    total_count = np.bincount(class_y)
    probabilities = total_count[np.nonzero(total_count)] / len(class_y)
    if len(probabilities) <= 1:
        return 0
    return -np.sum(probabilities * np.log(probabilities)) / np.log(len(probabilities))

def information_gain(previous_y, current_y):
    """
    Inputs:
    - previous_y : the distribution of original labels (0's and 1's)
    - current_y  : the distribution of labels after splitting
                   based on a particular split attribute and split value

    Output:
    - info_gain  : The information gain after partitioning

    Computes and returns the information gain from partitioning the previous_y
    labels into the current_y labels.
    """
    conditional_entropy = 0
    for y in current_y:
        conditional_entropy += (entropy(y) * len(y) / len(previous_y))
    info_gain = entropy(previous_y) - conditional_entropy
    return info_gain

def partition_classes(x,y,split_attribute,split_val):
    """
    Function to split dataset into left and right subsets
    based on a given split attribute and split value.

    Works for both numerical and categorical attributes.
    """
    X = np.array(x)
    column_split = X[:, split_attribute]
    X_left, X_right = [], []
    y_left, y_right = [], []
    counter = 0
    if isinstance(split_val, str) == False:
        for i in column_split:
            if i <= split_val:
                X_left.append(X[counter])
                y_left.append(y[counter])
            else:
                X_right.append(X[counter])
                y_right.append(y[counter])
            counter += 1
    else:
        for i in column_split:
            if i == split_val:
                X_left.append(X[counter])
                y_left.append(y[counter])
            else:
                X_right.append(X[counter])
                y_right.append(y[counter])
            counter += 1
    return X_left, X_right, y_left, y_right

def find_best_split(X, y, split_attribute):
    """
    Function to find the best split value for a given attribute.

    It works by:
    1. Checking all possible unique values of the chosen attribute.
    2. Partitioning the dataset based on each candidate split.
    3. Calculating the information gain for each split.
    4. Selecting the split value that gives the highest information gain.
    """
    best_info_gain = 0
    X = np.array(X)
    column_split = X[:, split_attribute]
    column_split = np.unique(column_split)
    best_split_val = column_split[0]
    for split_val in column_split:
        current_X_left, current_X_right, current_y_left, current_y_right = partition_classes(X, y, split_attribute, split_val)
        current_y = [current_y_left, current_y_right]
        current_info_gain = information_gain(y, current_y)
        if current_info_gain > best_info_gain:
            best_info_gain = current_info_gain
            best_split_val = split_val
    return best_split_val, best_info_gain

def find_best_feature(X, y):
    """
    Function to find the best feature (column) and its best split value
    for building a decision tree.

    Steps:
    1. Loop over all features (columns).
    2. For each feature, find the best split value and its info gain.
    3. Keep track of the feature that gives the highest info gain.
    4. Return the best feature index and split value.
    """
    best_info_gain = 0
    best_feature = 0
    best_split_val = 0
    for feature_index in range(len(X[0])):
        current_best_split_val, current_best_info_gain = find_best_split(X, y, feature_index)
        if current_best_info_gain > best_info_gain:
            best_info_gain = current_best_info_gain
            best_feature = feature_index
            best_split_val = current_best_split_val
    return best_feature, best_split_val

class MyDecisionTree(object):
  def __init__(self,max_depth=None):
    self.tree={}
    self.residual_tree={}
    self.max_depth=max_depth

  def fit(self, X, y, depth):
    """
    Build a decision tree recursively.

    Parameters:
    -----------
    X : dataset (features/attributes)
    y : labels (target classes)
    depth : current depth of the tree

    Returns:
    --------
    A decision tree node (either a class label for leaf nodes
    or a dictionary representing a decision rule with subtrees).
    """
    unique_labels = np.unique(y)
    if (len(unique_labels) == 1) or (depth == self.max_depth):
        unique_labels, counts_unique_labels = np.unique(y, return_counts=True)
        index = counts_unique_labels.argmax()
        classification = unique_labels[index]
        return classification
    best_feat, best_split = find_best_feature(X, y)
    X_left, X_right, y_left, y_right = partition_classes(X, y, best_feat, best_split)
    if isinstance(best_split, str):
        question = "{} == {}".format(best_feat, best_split)
    else:
        question = "{} <= {}".format(best_feat, best_split)
    node = {question: []}
    depth+=1
    yes_answer=self.fit(X_left,y_left,depth)
    no_answer=self.fit(X_right,y_right,depth)
    if yes_answer==no_answer:
      node=yes_answer
    else:
      node[question].append(yes_answer)
      node[question].append(no_answer)
    self.tree=node
    return node

  def predict(self, record, flag=1):
    """
    Predict the class label for a given record using the decision tree.

    Args:
        record: Input data point (list/array) to classify
        flag: Internal flag to track first call (1) vs recursive calls (0)

    Returns:
        Predicted class label
    """
    if flag == 1:
        self.residual_tree = self.tree
    question = list(self.residual_tree.keys())[0]
    feature, comparison, value = question.split()
    if comparison == "==":
        if record[int(feature)] == value:
            answer = self.residual_tree[question][0]
        else:
            answer = self.residual_tree[question][1]
    elif comparison == "<=":
        if float(record[int(feature)]) <= float(value):
            answer = self.residual_tree[question][0]
        else:
            answer = self.residual_tree[question][1]
    if not isinstance(answer, dict):
        return answer
    else:
        self.residual_tree = answer
        return self.predict(record, 0)

def DecisionTreeEvaluation(id3, X, y, verbose=False):
    """
    Evaluate the performance of a decision tree model by calculating accuracy.

    Args:
        id3: Trained decision tree model with predict method
        X: Test/validation feature data (list of records)
        y: True labels for the test data
        verbose: If True, prints the accuracy score

    Returns:
        accuracy: Float value representing the accuracy score (0 to 1)
    """
    y_predicted = []
    for record in X:
        y_predicted.append(id3.predict(record))
    results = []
    for prediction, truth in zip(y_predicted, y):
        results.append(prediction == truth)
    accuracy = float(results.count(True)) / float(len(results))
    if verbose:
        print('Accuracy: %.4f' % accuracy)
    return accuracy
