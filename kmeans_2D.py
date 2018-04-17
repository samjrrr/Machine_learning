"""
Created on Sun Apr  8 18:51:43 2018
kmeans model
@author: samrichardson
"""

import numpy as np #Used to handle arrays
import matplotlib.pyplot as plt

"""

Kmeans model 

Operation
Setup: kmeans class with arguments k (int) and initial clusters (numpy array)
Fit: Use self.fit method with data argument
Plot: self.plot with data and labels from fit method
Assigned labels and Sum of squared error can be called once model has been fit
    using self.labels and self.SSE respectively

"""

class K_means(): 
    
    #Intialise kmeans instance
    def __init__(self, k):
        self.k = k
             
    #Fit data using k means algorithm
    #Function repeats two steps until convergance is reached
    #1: Provides list of nearest cluster for each data point
    #2: Recalculates cluster centre points based on group mean
    #Convergance is reached when the sum of squared errors reaches equilibrium
    def fit(self, data):
        error2=0
        i = 0
        self.clusters = self.initial_clusters(data)
        while True:
            error1 = error2 
            self.labels, error2 = self.data_label_error(data)
            self.clusters = self.new_clusters(data, self.labels)
            if error1 == error2:
                print("Number of unique iterations:", (i-1))
                print("Squared standard error: {:.3f}".format(error2))
                print("Final cluster coordinates:\n", self.clusters)
                self.SSE = error1
                self.labels = self.labels
                break
            i += 1
    
    #Meansure distance between two sets of coordinates   
    def euc(self, a, b): 
        return np.linalg.norm(a - b)
          
    #Function to assign labels to data for given clusters
    #Returns list of assigned labels and sum of squared error          
    def data_label_error(self, data): 
        self.labels = []   
        SSE_list = []
        for row in data:        
            row_D_list = []     
            for i in range(self.k): 
                D = self.euc(row, self.clusters[i])
                #print(D)
                #D2 = self.dist(row, self.clusters[i])
                #print(D2)
                row_D_list.append(D) 
            min_value = min(row_D_list) 
            label = row_D_list.index(min_value)
            self.labels.append(label)
            SSE = min_value ** 2
            SSE_list.append(SSE)
        SSE_sum = sum(SSE_list)
        return np.array(self.labels), SSE_sum
    
    #Initialise clusters with random position 
    def initial_clusters(self, data):
        dimensions = len(data[0,:])
        max_x = max(data[:, 0])
        max_y = max(data[:, 1])
        self.initial_clusters = np.random.rand(self.k, dimensions)
        #Normalise postions based on max x, y values
        self.initial_clusters[:, 0] *= max_x
        self.initial_clusters[:, 1] *= max_y
        return self.initial_clusters

    #Establish new clusters from group mean
    def new_clusters(self, data, labels):    
        new_clusters_x = []
        new_clusters_y = []
        for i in range(self.k):
            self.clusters = np.empty([self.k, 2])
            df_cluster_i = data[labels == i]
            mean_x = np.mean(df_cluster_i[:,0])
            mean_y = np.mean(df_cluster_i[:,1])        
            new_clusters_x.append(mean_x)
            new_clusters_y.append(mean_y)   
        self.clusters = np.column_stack((new_clusters_x, new_clusters_y))
        return self.clusters
    
    def plot2D(self, data, labels):
        data_length = len(data)
        column_reshape = np.array(labels).reshape(data_length,1)
        data_to_plot = np.hstack((data, column_reshape))
        plt.subplots(figsize=(5,5))
        plt.rcParams.update({'font.size': 14})
        for i in range(self.k):
            zero = data_to_plot[data_to_plot[:, 2] == i]
            plt.scatter(zero[:,0], zero[:,1], s = 10, label = i)
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.show()
            
"""

Model fit with random data 

"""

#Set random seed
np.random.seed(seed=1)

#Create random data
random_data = np.random.rand(10000, 2)
    
#Create instance of model            
kmeans = K_means(5)

#Fit model to example data
kmeans.fit(random_data)

#Extract list of labels from fitted model
labels = kmeans.labels

#Plot data
kmeans.plot2D(random_data, labels)
