import heapq
import sys

N, K, Q = map(int, sys.stdin.readline().split())
A = [0]*N
heap = []

for _ in range(Q):
    X, Y = map(int, sys.stdin.readline().split())
    A[X-1] = Y
    heapq.heappush(heap, (-Y, X-1))
    while len(heap) > K:
        heapq.heappop(heap)
    print(-sum(y for y, x in heap))