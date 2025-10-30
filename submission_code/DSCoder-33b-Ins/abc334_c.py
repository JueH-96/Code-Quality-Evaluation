# YOUR CODE HERE

import sys

def min_weirdness():
    N, K = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))

    colors = [0] * (N+1)
    for i in A:
        colors[i] = 1

    left = [0] * (N+1)
    right = [0] * (N+1)
    for i in range(1, N+1):
        if colors[i] == 0:
            left[i] = left[i-1] + 1
        else:
            left[i] = 0

    for i in range(N, 0, -1):
        if colors[i] == 0:
            right[i] = right[i+1] + 1
        else:
            right[i] = 0

    total_weirdness = 0
    for i in range(1, N+1):
        if colors[i] == 0:
            total_weirdness += min(left[i], right[i])

    return total_weirdness

print(min_weirdness())