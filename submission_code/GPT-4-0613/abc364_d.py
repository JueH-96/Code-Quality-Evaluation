import bisect
import sys
input = sys.stdin.readline

N, Q = map(int, input().split())
A = sorted(list(map(int, input().split())))

for _ in range(Q):
    B, K = map(int, input().split())
    idx = bisect.bisect_left(A, B)
    if idx < K:
        print(max(B - A[idx - K], A[idx + K - 1] - B))
    else:
        print(B - A[idx - K])