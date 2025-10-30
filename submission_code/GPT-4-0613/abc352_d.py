import sys
from heapq import heappush, heappop

def solve():
    N, K = map(int, sys.stdin.readline().split())
    P = list(map(int, sys.stdin.readline().split()))
    P = [(p, i) for i, p in enumerate(P)]
    P.sort()
    heap = []
    res = float('inf')
    for p, i in P:
        heappush(heap, i)
        if len(heap) > K:
            heappop(heap)
        if len(heap) == K:
            res = min(res, max(heap) - min(heap) + 1)
    print(res)

solve()