# YOUR CODE HERE

import sys

def solve():
    N, M = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    B = list(map(int, sys.stdin.readline().split()))

    C = sorted(A + B)

    for i in range(len(C) - 1):
        if C[i] == C[i + 1] - 1:
            print("Yes")
            return
    print("No")

solve()