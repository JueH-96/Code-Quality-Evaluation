# YOUR CODE HERE
def min_tour_length(N, M, X):
    # Calculate the total distance without closing any bridge
    total_distance = 0
    for i in range(1, M):
        total_distance += min(abs(X[i] - X[i-1]), N - abs(X[i] - X[i-1]))
    
    # Calculate the distance saved by closing each bridge
    max_saving = 0
    for i in range(1, M):
        direct_distance = abs(X[i] - X[i-1])
        alternative_distance = N - direct_distance
        saving = abs(direct_distance - alternative_distance)
        max_saving = max(max_saving, saving)
    
    # The minimum possible length of the tour
    min_length = total_distance - max_saving
    return min_length

import sys
input = sys.stdin.read
data = input().split()
N = int(data[0])
M = int(data[1])
X = list(map(int, data[2:]))

print(min_tour_length(N, M, X))