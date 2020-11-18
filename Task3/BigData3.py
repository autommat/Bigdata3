import findspark
findspark.init()
from pyspark import SparkContext, SparkConf
from typing import List,Callable
import matplotlib.pyplot as plt

#CONSTANT PARAMETERS
MAX_ITERATIONS=20
CLUSTERS=10
RANDOM_INIT_CENTR_FILENAME='3b.txt'
FAR_INIT_CENTR_FILENAME='3c.txt'

def euclid_dist(point, cluster_centr):
    sum = 0
    for p, c in zip(point, cluster_centr):
        sum += (p-c)**2
    return sum**0.5

def euclid_phi_cost(point, cluster_centr):
    return euclid_dist(point, cluster_centr)**2

def manhattan_dist(point, cluster_centr):
    sum = 0
    for p,c in zip(point, cluster_centr):
        sum += abs(p-c)
    return sum

def manhattan_psi_cost(point, cluster_centr):
    return manhattan_dist(point, cluster_centr)

def assign_to_clust(point, centroids):
    clust_index = None
    min_dist = None
    for clust_num, clust_centr in enumerate(centroids):
        dist = calculate_dist(point, clust_centr)
        if min_dist is None or dist < min_dist:
            min_dist = dist
            clust_index = clust_num
    return clust_index, point

def avg(points):
    point_len = None
    sum = None
    how_many = 0
    for point in points:
        how_many+=1
        if point_len is None:
            point_len = len(point)
            sum = point_len*[0]
        for i,each in enumerate(point):
            sum[i]+=each
    return [each/how_many for each in sum]

# program

calculate_dist: Callable = euclid_dist
calculate_cost: Callable = euclid_phi_cost
init_centroid_file: str = RANDOM_INIT_CENTR_FILENAME
out_filename=""

dist_choice = input("select: 1 for euclidean, 2 for manhattan distance")
if dist_choice == '1':
    calculate_dist = euclid_dist
    calculate_cost = euclid_phi_cost
    out_filename+="euclidean"
elif dist_choice =='2':
    calculate_dist = manhattan_dist
    calculate_cost = manhattan_psi_cost
    out_filename+="manhattan"
else:
    print('wrong input')
    exit(1)

file_choice = input("select: 1 for random, 2 for far initial centers")
if file_choice == '1':
    init_centroid_file = RANDOM_INIT_CENTR_FILENAME
    out_filename+="_random"
elif file_choice == '2':
    init_centroid_file = FAR_INIT_CENTR_FILENAME
    out_filename+="_far"
else:
    print('wrong input')
    exit(1)

centroids:List[List[float]] = []
with open(init_centroid_file, "r") as centr_file:
    for line in centr_file.readlines():
        centroids.append(list(map(float, line.split())))

sc = SparkContext(conf=SparkConf())
data_from_file = sc.textFile('3a.txt')
points = data_from_file.map(lambda line: list(map(float, line.split())))

cost_at_iter=[]
for iter in range(MAX_ITERATIONS):
    clusts_to_points = points.map(lambda point: assign_to_clust(point, centroids))
    cost = clusts_to_points.map(lambda clust_to_point: calculate_cost(clust_to_point[1], centroids[clust_to_point[0]])).sum()
    cost_at_iter.append(cost)
    clusts_to_all_points = clusts_to_points.groupByKey()
    clusts_to_mean = clusts_to_all_points.mapValues(avg)
    centroids = clusts_to_mean.map(lambda clust_to_mean: clust_to_mean[1]).collect()

with open(f'{out_filename}.txt', 'w') as out_file:
    for i, cost in enumerate(cost_at_iter):
        out_file.write(f'{i}\t{cost}\n')
    out_file.write(f'{100*(cost_at_iter[9]-cost_at_iter[0])/cost_at_iter[0]}%')

plt.title(out_filename)
plt.plot(cost_at_iter, marker='o')
plt.savefig(out_filename+".jpg")