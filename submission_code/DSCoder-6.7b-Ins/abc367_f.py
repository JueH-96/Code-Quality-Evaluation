import sys
from collections import Counter

def solve():
    N, Q = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    B = list(map(int, sys.stdin.readline().split()))
    queries = [list(map(int, sys.stdin.readline().split())) for _ in range(Q)]

    for l, r, L, R in queries:
        A_sub = A[l-1:r]
        B_sub = B[L-1:R]
        if Counter(A_sub) == Counter(B_sub):
            print("Yes")
        else:
            print("No")

solve()