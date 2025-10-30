import sys

def read_input():
    N, Q = map(int, sys.stdin.readline().split())
    P = [sys.stdin.readline().strip() for _ in range(N)]
    queries = [list(map(int, sys.stdin.readline().split())) for _ in range(Q)]
    return N, Q, P, queries

def solve(N, Q, P, queries):
    prefix_sum = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(N):
        for j in range(N):
            prefix_sum[i + 1][j + 1] = prefix_sum[i + 1][j] + prefix_sum[i][j + 1] - prefix_sum[i][j] + (P[i][j] == 'B')
    for A, B, C, D in queries:
        print(prefix_sum[C % N][D % N] - prefix_sum[A - 1][D % N] - prefix_sum[C % N][B - 1] + prefix_sum[A - 1][B - 1])

N, Q, P, queries = read_input()
solve(N, Q, P, queries)