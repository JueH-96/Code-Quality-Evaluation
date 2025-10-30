import sys

def solve():
    N, Q = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    mex = [0] * (max(A) + 1)
    for a in A:
        mex[a] += 1
    for _ in range(Q):
        i, x = map(int, sys.stdin.readline().split())
        mex[x] += 1
        print(mex.index(0))

solve()