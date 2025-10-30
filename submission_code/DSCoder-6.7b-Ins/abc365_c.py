import sys

def max_subsidy_limit(N, M, A):
    A.sort()
    for i in range(N):
        if A[i] * (N - i) <= M:
            M -= A[i] * (N - i)
        else:
            return M // (N - i)
    return 'infinite'

N, M = map(int, sys.stdin.readline().split())
A = list(map(int, sys.stdin.readline().split()))

print(max_subsidy_limit(N, M, A))