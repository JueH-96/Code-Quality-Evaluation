# YOUR CODE HERE
import sys

def min_max_diff(N, K, A):
    A.sort()
    min_diff = float('inf')
    
    for i in range(N - K + 1):
        diff = A[i + K - 1] - A[i]
        min_diff = min(min_diff, diff)
    
    return min_diff

input = sys.stdin.read
data = input().split()

N = int(data[0])
K = int(data[1])
A = list(map(int, data[2:]))[:N]

print(min_max_diff(N, K, A))