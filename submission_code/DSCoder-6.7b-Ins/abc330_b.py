import sys

def solve():
    N, L, R = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))

    for a in A:
        if a < L:
            print(L, end=' ')
        elif a > R:
            print(R, end=' ')
        else:
            print(a, end=' ')

solve()