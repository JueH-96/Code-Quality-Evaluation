# YOUR CODE HERE

import sys

def solve():
    N, A = map(int, sys.stdin.readline().split())
    T = list(map(int, sys.stdin.readline().split()))

    time = 0
    for i in range(N):
        if T[i] >= time:
            time = T[i] + A
        else:
            time += A
        print(time)

solve()