import sys

def solve():
    input = sys.stdin.read
    N, K, *A = map(int, input().split())
    A.sort()
    min_diff = float('inf')
    for i in range(N - K):
        diff = A[i + K - 1] - A[i]
        min_diff = min(min_diff, diff)
    print(min_diff)

solve()