import pandas as pd
import numpy as np
import csv
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score

input_data=pd.read_csv("../data/2_98/test.csv", header=None, sep="@", names=['label', 'sentence'])
mapping = {'Liberal': 0, 'Conservative': 1, 'Neutral': 2}
output = input_data.replace({'label': mapping})
true = output['label'].tolist()
with open('98_2_750eppre.csv', 'w', newline='') as f:
    csvwriter = csv.writer(f, delimiter=';')
    csvwriter.writerow(["ep", "f1", "prec", "recall", "accuracy"])
    for i in range(0,8):
        results_data=pd.read_csv("../old_output/98_2_750eppre/results_ep" + str(i) + ".txt", header=None, names=['label'])

        res = results_data['label'].tolist()

        f1 = f1_score(true, res, average=None)
        precision = precision_score(true, res, average=None)
        recall = recall_score(true, res, average=None)
        accuracy = accuracy_score(true, res)
        print("ep" + str(i) + ": f1: " + str(f1) + " precision: " + str(precision) + " recall: " + str(recall))


        csvwriter.writerow(["ep" + str(i),np.mean(f1),np.mean(precision),np.mean(recall),np.mean(accuracy)])