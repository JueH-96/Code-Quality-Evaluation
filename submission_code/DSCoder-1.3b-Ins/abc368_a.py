# YOUR CODE HERE

import sys

def solve():
    N, K = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))

    stack = A[N-K:N] + A[:N-K]

    for i in stack:
        print(i, end=' ')

solve()