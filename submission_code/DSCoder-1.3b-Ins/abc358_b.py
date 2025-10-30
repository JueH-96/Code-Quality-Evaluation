# YOUR CODE HERE

import sys

def solve():
    N, A = map(int, sys.stdin.readline().split())
    T = list(map(int, sys.stdin.readline().split()))

    T.sort()

    for i in range(N):
        if i == 0:
            print(max(0, T[i] - A))
        else:
            print(max(0, T[i] - T[i-1] - A))

solve()