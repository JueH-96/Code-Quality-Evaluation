import sys
from bisect import bisect_right

def solve():
    N, M = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))

    for i in range(1, N+1):
        idx = bisect_right(A, i)
        if idx == M:
            print(0)
        else:
            print(A[idx] - i)

solve()