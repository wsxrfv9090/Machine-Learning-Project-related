import numpy as np
from math import sqrt
import warnings
from matplotlib import style
from collections import Counter
import pandas as pd
import random
import file_operations
import os

def k_nearest_neighbor(data, predict, k = 3):
    if len(data) >= k:
        warnings.warn('K is set to a value less than total voting groups! Idiot!')
    distances = []
    for group in data:
        for features in data[group]:
            euclidean_distance = np.linalg.norm(np.array(features) - np.array(predict))
            distances.append([euclidean_distance, group])
            
    votes = [i[1] for i in sorted(distances)[:k]]
    vote_result = Counter(votes).most_common(1)[0][0]
    
    confidence = Counter(votes).most_common(1)[0][1] / k
    
    return vote_result, confidence

accuracies = []

current_dir = os.getcwd()
paths = file_operations.create_project_path('Project 5 KNN')
file_name = 'breast-cancer-wisconsin.data'
file_path = os.path.join(paths["data_dir"], file_name)

for i in range(10000):

    df = pd.read_csv(file_path)

    df.replace('?', np.nan, inplace = True)
    df.dropna(inplace = True)
    # df.replace('?', -99999, inplace = True)
    df.drop(['id'], axis = 1, inplace = True)

    full_data = df.astype('float64').values.tolist()

    random.shuffle(full_data)

    test_size = 0.2
    train_set = {2: [], 4: []}
    test_set = {2: [], 4: []}
    delimeter = -int(test_size * len(full_data))
    train_data = full_data[:delimeter]
    test_data = full_data[delimeter:]

    for i in train_data:
        train_set[i[-1]].append(i[:-1])
        
    for i in test_data:
        test_set[i[-1]].append(i[:-1])
        
    correct = 0
    total = 0

    for group in test_set:
        for data in test_set[group]:
            vote, confidence2 = k_nearest_neighbor(train_set, data, k = 5)
            if group == vote:
                correct += 1
            total += 1
    
    accuracy = correct/total   
    # print(accuracy)
    accuracies.append(accuracy)
    
print(sum(accuracies)/len(accuracies))