import sys

def solve(N, M, C, K, A):
    result = 0
    for k in range(K):
        min_val = min(A[i] for i in range(N) if (C+k)%M == A[i]%M)
        result += min_val
    return result

N, M, C, K = map(int, sys.stdin.readline().split())
A = list(map(int, sys.stdin.readline().split()))

print(solve(N, M, C, K, A))