from collections import defaultdict
import numpy as np
import random
from decision_tree import DecisionTree

class RandomForest:
    def __init__(self, params):
        self.forest = []
        self.params = defaultdict(lambda: None, params)


    def train(self, X, y):
        for _ in range(self.params["ntrees"]):
            X_bagging, y_bagging = self.bagging(X,y)
            tree = DecisionTree(self.params)
            tree.train(X_bagging, y_bagging)
            self.forest.append(tree)

    def evaluate(self, X, y):
        predicted = self.predict(X)
        predicted = [round(p) for p in predicted]
        print(f"Accuracy: {round(np.mean(predicted==y),2)}")

    def predict(self, X):
        tree_predictions = []
        for tree in self.forest:
            tree_predictions.append(tree.predict(X))
        forest_predictions = list(map(lambda x: sum(x)/len(x), zip(*tree_predictions)))
        return forest_predictions

    def bagging(self, X, y):
        X_selected, y_selected = None, None
        #X_subarray=X[int(0.63*len(X)):]
        #y_subarray=y[int(0.63*len(y)):]
        #X_selected=random.choice(X_subarray)
        #y_selected=random.choice(y_subarray)
        index_selected=np.random.choice(len(X),len(X),True)
        X_selected=X[index_selected]
        y_selected=y[index_selected]
        # TODO implement bagging


        return X_selected, y_selected
