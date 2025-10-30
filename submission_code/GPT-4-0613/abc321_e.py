import sys
from math import log2

def solve():
    T = int(sys.stdin.readline())
    for _ in range(T):
        N, X, K = map(int, sys.stdin.readline().split())
        if K <= log2(X):
            print(2**K)
        elif K <= log2(N+1):
            print(min(2**K, 2**(log2(N+1)-K+1)-1))
        else:
            print(0)

solve()