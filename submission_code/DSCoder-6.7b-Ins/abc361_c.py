import sys

def solve():
    N, K = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))

    A.sort()

    print(min(A[i+K-1]-A[i] for i in range(N-K+1)))

solve()