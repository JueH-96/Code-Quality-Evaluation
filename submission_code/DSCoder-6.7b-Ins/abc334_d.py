import sys
import bisect

def read_input():
    N, Q = map(int, sys.stdin.readline().split())
    R = list(map(int, sys.stdin.readline().split()))
    queries = [int(sys.stdin.readline()) for _ in range(Q)]
    return N, Q, R, queries

def solve(N, Q, R, queries):
    R.sort()
    prefix_sum = [0] * (N+1)
    for i in range(1, N+1):
        prefix_sum[i] = prefix_sum[i-1] + R[i-1]

    for query in queries:
        idx = bisect.bisect_right(prefix_sum, query)
        sys.stdout.write(str(idx) + '
')

N, Q, R, queries = read_input()
solve(N, Q, R, queries)