# YOUR CODE HERE
from collections import defaultdict
from heapq import heappop, heappush

def solve():
    import sys
    input = sys.stdin.readline

    N, M = map(int, input().split())
    edges = defaultdict(list)
    for _ in range(M):
        u, v = map(int, input().split())
        edges[u-1].append(v-1)
        edges[v-1].append(u-1)
    W = list(map(int, input().split()))
    A = list(map(int, input().split()))

    pq = []
    for i in range(N):
        for _ in range(A[i]):
            heappush(pq, (-W[i], i))

    ans = 0
    while pq:
        w, x = heappop(pq)
        w = -w
        S = []
        for y in edges[x]:
            if W[y] < w:
                S.append(y)
        for y in S:
            A[y] += 1
            heappush(pq, (-W[y], y))
        ans += 1

    print(ans)

solve()