# YOUR CODE HERE
from heapq import *

N, K, Q = map(int, input().split())
A = [0] * N
heap = []
for i in range(Q):
    X, Y = map(int, input().split())
    X -= 1
    if A[X] > 0:
        heappush(heap, -A[X])
    A[X] = Y
    if len(heap) < K:
        heappush(heap, -A[X])
    elif A[X] > -heap[0]:
        heappushpop(heap, -A[X])
    print(-sum(heap))