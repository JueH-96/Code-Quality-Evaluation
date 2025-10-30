import sys

def solve():
    N, K = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))

    total = sum(range(1, K+1))
    present = set(A)

    missing = total - sum(present)
    print(missing)

solve()