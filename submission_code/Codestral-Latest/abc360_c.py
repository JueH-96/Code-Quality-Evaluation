import sys
from collections import defaultdict

input = sys.stdin.read
data = input().split()

N = int(data[0])
A = list(map(int, data[1:N+1]))
W = list(map(int, data[N+1:]))

# Dictionary to store the total weight of items in each box
box_weights = defaultdict(int)
for i in range(N):
    box_weights[A[i]] += W[i]

# Sort the weights in ascending order
sorted_weights = sorted(box_weights.values())

# Calculate the minimum cost
min_cost = 0
for i in range(1, len(sorted_weights)):
    min_cost += sorted_weights[i]

print(min_cost)