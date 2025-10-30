import sys
from sortedcontainers import SortedSet

def mex(A):
    A = SortedSet(A)
    mex = 0
    for a in A:
        if a == mex:
            mex += 1
        elif a > mex:
            return mex
    return mex + (A[-1] < mex)

def solve():
    N, Q = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    queries = [list(map(int, sys.stdin.readline().split())) for _ in range(Q)]

    for i, x in queries:
        A[i-1] = x
        print(mex(A))

# Call the function
solve()