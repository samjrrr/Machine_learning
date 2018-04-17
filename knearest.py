import numpy as np
from sklearn import datasets
from scipy.spatial import distance #Used to calculate euclidean distance
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from scipy.stats import mode

"""
Knn classifier model
Useage: initilise knn model with arguments number of neighbors (n) and algorithm
Algorithms: classifier, regression
Fit: self.fit w/ train y and x
Predict: self.predict w/ test data  
"""

class knn():
    #Initialise model
    def __init__(self, n = 5, algorithm = "classifier"):
        self.n = n
        self.algorithm = algorithm
    #Fit model by copying train data to class instance    
    def fit(self, train_X, train_y):
        self.train_X = train_X
        self.train_y = train_y
    #Function to determine n nearest neighbors and find either mode or mean   
    def predict(self, test_X):
        average_tagets = []
        labels = self.best_index(test_X)
        for row in labels:
            row_label = self.train_y[row]
            if self.algorithm == "regression":
                average_value = np.mean(row_label)
                average_tagets.append(average_value)
            if self.algorithm == "classifier":
                average_value = mode(row_label, axis=0)[0]
                average_tagets.append(average_value)
        return average_tagets
        
    #Measure distance between two sets of coordinates   
    def euc(self, a, b): 
        return distance.euclidean(a,b)
    
    #Return list of n nearest targets for test data            
    def best_index(self, test_X):
        #Function currently calculates distance between each train and test point
        #Returns 5 lowest indexs
        i = 0
        test_length = len(test_X)
        train_length = len(self.train_X)
        distance_array = np.zeros(shape=(test_length, train_length))
        min_distances = []
        for i in range(len(test_X)):
            j = 0
            for j in range(len(self.train_X)):
                distance_array[i,j] = self.euc(self.train_X[j], test_X[i])
            distance_array_sort = distance_array[i].argsort()[:self.n]
            min_distances.append(distance_array_sort)
            min_distance_labeled = min_distances
        return min_distance_labeled  
        
"""
Model Example
"""    
#Load example data from scikitlearn
iris = datasets.load_iris()
X = iris.data
y = iris.target

#Split into training and test sets
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.2)

#Create model instance
my_knn = knn(5, "classifier")

#Fit model using training data
my_knn.fit(train_X, train_y)
pred_y = my_knn.predict(test_X)

#Test model accuracy
print("Prediction accuracy: {:.3f}".format(accuracy_score(test_y, pred_y)))
