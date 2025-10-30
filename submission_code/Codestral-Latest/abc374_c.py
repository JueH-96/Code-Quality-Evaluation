import sys
from itertools import combinations

input = sys.stdin.read
data = input().split()

N = int(data[0])
K = list(map(int, data[1:]))

# Calculate the total number of people
total = sum(K)

# Initialize the minimum possible value of the maximum number of people taking a lunch break at the same time
min_max = float('inf')

# Iterate through all possible ways to divide the departments into two groups
for i in range(1, N):
    for combo in combinations(K, i):
        group_a = sum(combo)
        group_b = total - group_a
        min_max = min(min_max, max(group_a, group_b))

print(min_max)