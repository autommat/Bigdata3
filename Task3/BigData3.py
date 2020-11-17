from statistics import mean

import findspark
findspark.init()
from pyspark import SparkContext, SparkConf
from typing import *

def euclid_dist(point, cluster_centr):
    sum = 0
    for p, c in zip(point, cluster_centr):
        sum += (p-c)**2
    return sum**0.5

def manhattan_dist(point, cluster_centr):
    sum = 0
    for p,c in zip(point, cluster_centr):
        sum += abs(p-c)
    return sum

def euclid_phi_cost(point, cluster_centr):
    return euclid_dist(point, cluster_centr)**2

def manhattan_psi_cost(point, cluster_centr):
    return manhattan_dist(point, cluster_centr)

MAX_ITERATIONS=20
CLUSTERS=10
RANDOM_INIT_CENTR_FILENAME='3b.txt'
FAR_INIT_CENTR_FILENAME='3c.txt'

calculate_dist: Callable = euclid_dist
calculate_cost: Callable = euclid_phi_cost


centroids:List[List[float]] = []

with open(RANDOM_INIT_CENTR_FILENAME, "r") as centr_file:
    centroids.append(list(map(float, centr_file.readline().split())))


sc = SparkContext(conf=SparkConf())
data_from_file = sc.textFile('3a.txt')
points = data_from_file.map(lambda line: list(map(float, line.split())))

# peek = data_as_vector.take(10)
# print(peek)



def assign_to_clust(point, centroids):
    clust_index = None
    min_dist = None
    for clust_num, clust_centr in enumerate(centroids):
        dist = calculate_dist(point, clust_centr)
        if min_dist is None or dist < min_dist:
            min_dist= dist
            clust_index = clust_num
    return clust_index, point

def avg(points):
    return list(map(mean, zip(*points)))
    # how_many =None
    # sum=None
    # for point in points:
    #     if how_many is None:
    #         how_many = len(point)
    #         sum=how_many*[0]
    #     for i,each in enumerate(point):
    #         sum[i]+=each
    # return [each/how_many for each in sum]


cost_at_iter=[]
for iter in range(MAX_ITERATIONS):
    clusts_to_points = points.map(lambda point: assign_to_clust(point, centroids))
    cost = clusts_to_points.map(lambda clust_to_point: calculate_cost(clust_to_point[1], centroids[clust_to_point[0]])).sum()
    cost_at_iter.append(cost)
    clusts_to_all_points = clusts_to_points.groupByKey()
    print(clusts_to_all_points.take(1))
    clusts_to_mean = clusts_to_all_points.mapValues(avg)
    centroids = clusts_to_mean.map(lambda clust_to_mean: clust_to_mean[1]).collect()
    print(centroids)
print(cost_at_iter)
