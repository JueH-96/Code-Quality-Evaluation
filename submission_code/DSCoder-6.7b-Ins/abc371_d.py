import sys
from bisect import bisect_left, bisect_right

def read_input():
    N = int(sys.stdin.readline().strip())
    X = list(map(int, sys.stdin.readline().strip().split()))
    P = list(map(int, sys.stdin.readline().strip().split()))
    Q = int(sys.stdin.readline().strip())
    queries = [tuple(map(int, sys.stdin.readline().strip().split())) for _ in range(Q)]
    return N, X, P, Q, queries

def solve():
    N, X, P, Q, queries = read_input()
    villages = sorted(zip(X, P))
    prefix_sum = [0] * (N + 1)
    for i in range(N):
        prefix_sum[i + 1] = prefix_sum[i] + villages[i][1]
    for L, R in queries:
        left = bisect_left(X, L)
        right = bisect_right(X, R)
        print(prefix_sum[right] - prefix_sum[left])

solve()