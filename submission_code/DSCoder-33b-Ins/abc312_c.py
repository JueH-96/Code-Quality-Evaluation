# YOUR CODE HERE

import sys

def solve():
    N, M = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    B = list(map(int, sys.stdin.readline().split()))

    A.sort()
    B.sort()

    left = 0
    right = N

    while right - left > 1:
        mid = (left + right) // 2
        if A[mid] >= B[M - mid]:
            right = mid
        else:
            left = mid

    print(A[right])

solve()