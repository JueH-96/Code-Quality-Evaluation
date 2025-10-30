import sys

def solve():
    N, M, P = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    B = list(map(int, sys.stdin.readline().split()))

    A.sort()
    B.sort()

    total = 0
    for a in A:
        for b in B:
            total += min(a+b, P)

    print(total)

solve()