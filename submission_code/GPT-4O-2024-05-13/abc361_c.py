# YOUR CODE HERE
def min_max_difference(N, K, A):
    A.sort()
    min_diff = float('inf')
    for i in range(N - K):
        min_diff = min(min_diff, A[i + (N - K - 1)] - A[i])
    return min_diff

import sys
input = sys.stdin.read
data = input().split()
N = int(data[0])
K = int(data[1])
A = list(map(int, data[2:]))

print(min_max_difference(N, K, A))