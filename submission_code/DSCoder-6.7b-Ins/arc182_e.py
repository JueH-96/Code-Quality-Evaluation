import sys

def solve():
    N, M, C, K = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))

    min_values = [min(A[i] % M, (A[i] + C) % M) for i in range(N)]
    min_values.sort()

    total = 0
    for k in range(K):
        total += (C * k + min_values[k % N]) % M

    print(total)

solve()