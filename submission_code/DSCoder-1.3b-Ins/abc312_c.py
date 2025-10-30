import sys

def solve():
    N, M = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    B = list(map(int, sys.stdin.readline().split()))

    A.sort()
    B.sort(reverse=True)

    X = A[0]
    while True:
        if sum(x >= X for x in A) >= sum(x < X for x in B):
            return X
        X += 1

print(solve())