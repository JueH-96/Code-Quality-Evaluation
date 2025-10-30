# YOUR CODE HERE
from math import log2, floor

def count_vertices(N, X, K):
    if K == 0:
        return 1 if X == 1 else 0
    level = floor(log2(X))
    if level + K > floor(log2(N)):
        return 0
    return 2 ** K

T = int(input())
for _ in range(T):
    N, X, K = map(int, input().split())
    print(count_vertices(N, X, K))