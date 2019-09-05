import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import time
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from random import randint


class Gap:
    def __init__(self, n_cl_min=1, n_cl=21, nref=20, random_state=None):

        """Create a new Gap instance in orderd to computer gap statistics through Kmeans for a specified dataset
           Parameters
            ----------
            n_cl_min : int
                Minimum number of clusters for Gap computation
            metric : int
                Final clusters number
            nref : int
                Number of instances of random datasets
            random_state int, Random State instance or None:
                Determines random number generation for centroid initialization. Use
                an int to make the randomness deterministic. Default is None.

           Returns
            -------
            Gap instance : Object. """

        self.n_cl_min = n_cl_min
        self.n_cl = n_cl
        self.nref = nref
        self.random_state = random_state
        self.gap = np.array([])
        self.std_random = []

    def distance_rec(self, pos, n_clusters_, labels):

        """ Compute distance between each sample and its cluster center
            ----------
           pos : array-like or sparse matrix, shape=(n_samples, n_features)
                Training samples on which compute the distance
           n_clusters : array-like
                cluster centers
           labels : array-like, list
                centers labels

           Returns
            -------
            distance : array-like
                array of distances from sample points to centers. """

        distance = []
        frame = pd.DataFrame(pos)
        frame['labels'] = labels
        for name,group in frame.groupby('labels'):
            if(len(group) > 1):
                distance.append(pdist(group.values[:,:len(pos[1])]).sum()/len(group))
            else:
                distance.append(0)
        return distance

    def gap_statistic(self, X):

        """ Compute Gap statistics on a specified dataset
            Parameters
            ----------
            X : array-like or sparse matrix, shape=(n_samples, n_features)
                Training samples on which computer the gap statistics

           Returns
            -------
            Gaps, standard deviations : tuple of array-like, list objects. """

        if self.random_state == None:
            self.random_state=randint(1,1000)

        rands = (X.max()-X.min())*np.random.random_sample(size=(X.shape[0],X.shape[1], self.nref))+X.min()
        all_mean_data, all_mean_random, all_mean_random_gap = ([],[],[])
        for i in range(self.n_cl_min, self.n_cl):

            t1 = time.clock()
            mean_clusters = []
            km = KMeans(init='k-means++',n_clusters=i,random_state=self.random_state)
            km.fit(X)
            labels = km.labels_
            labels_unique = np.unique(labels)
            n_clusters_=len(labels_unique)
            mean_clusters = self.distance_rec(X, n_clusters_, labels)
            all_mean_data.append(np.sum(mean_clusters))

            rand_means = []

            for j in range(rands.shape[2]):
                mean_clusters = []
                km = KMeans(init='k-means++',max_iter=1000,n_clusters=i,random_state=self.random_state)
                km.fit(rands[:,:,j])
                labels = km.labels_
                labels_unique = np.unique(labels)
                n_clusters_=len(labels_unique)
                mean_clusters = self.distance_rec(rands[:,:,j], n_clusters_,labels)
                rand_means.append(np.sum(mean_clusters))

            all_mean_random.append(np.average(np.array(rand_means)))
            all_mean_random_gap.append(np.average(np.log(np.array(rand_means))))
            self.std_random.append(np.std(np.log(np.array(rand_means)))*np.sqrt(1+(1/self.nref)))
            print('Cicle number ' + str(i) + ' time: ' + str(time.clock()-t1))

        self.gap = np.array(all_mean_random_gap)-np.log(np.array(all_mean_data))

    def gap_max(self):
        if len(self.gap) == 0 or len(self.std_random) == 0:
            print("It's not possibile to calculate gap_max as no gap or std is present")
            return False

        for k in range(len(self.gap)):
            try:
                if(self.gap[k]>= self.gap[k+1] - self.std_random[k+1]):
                    return k + self.n_cl_min
            except IndexError:
                print('No max value has been found')
                return False
