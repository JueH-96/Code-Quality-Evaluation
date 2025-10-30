import sys
from bisect import bisect_right

input = sys.stdin.read
data = input().split()

N = int(data[0])
Q = int(data[1])

R = list(map(int, data[2:2+N]))
queries = list(map(int, data[2+N:]))

# Prefix sum of R
prefix_sum = [0] * (N + 1)
for i in range(N):
    prefix_sum[i+1] = prefix_sum[i] + R[i]

# Binary search to find the maximum number of sleighs that can be pulled with X reindeer
def max_sleighs(X):
    low, high = 0, N
    while low < high:
        mid = (low + high + 1) // 2
        if prefix_sum[mid] <= X:
            low = mid
        else:
            high = mid - 1
    return low

# Process each query
results = []
for X in queries:
    results.append(max_sleighs(X))

# Output the results
for result in results:
    print(result)