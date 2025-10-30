import sys

def solve():
    N, K = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))

    B = A[K:]
    B.sort()

    min_B = B[0]
    max_B = B[-1]

    return max_B - min_B

print(solve())