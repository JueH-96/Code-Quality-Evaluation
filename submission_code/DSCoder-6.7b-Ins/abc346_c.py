import sys

def solve():
    N, K = map(int, sys.stdin.readline().split())
    A = set(map(int, sys.stdin.readline().split()))

    missing = set(range(1, K+1)) - A
    print(sum(missing))

solve()