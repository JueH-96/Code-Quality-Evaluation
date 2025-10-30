import sys

def solve():
    N, K = map(int, sys.stdin.readline().split())
    X = list(map(int, sys.stdin.readline().split()))
    A = list(map(int, sys.stdin.readline().split()))

    for _ in range(K):
        B = [A[i] for i in X]
        A = B

    print(*A)

solve()