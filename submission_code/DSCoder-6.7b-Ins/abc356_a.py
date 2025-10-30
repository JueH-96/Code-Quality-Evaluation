# YOUR CODE HERE

import sys

N, L, R = map(int, sys.stdin.readline().split())
A = list(range(1, N+1))

A[L-1:R] = A[L-1:R][::-1]

print(' '.join(map(str, A)))