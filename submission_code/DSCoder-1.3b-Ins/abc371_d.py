import sys

def solve():
    N = int(sys.stdin.readline().strip())
    X = list(map(int, sys.stdin.readline().strip().split()))
    P = list(map(int, sys.stdin.readline().strip().split()))
    Q = int(sys.stdin.readline().strip())
    LR = [list(map(int, sys.stdin.readline().strip().split())) for _ in range(Q)]

    # Pre-calculate the prefix sum
    prefix_sum = [0] * (N + 1)
    for i in range(N):
        prefix_sum[i + 1] = prefix_sum[i] + P[i]

    # Process the queries
    for L, R in LR:
        print(prefix_sum[R] - prefix_sum[L - 1])

solve()