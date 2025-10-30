from heapq import heappush, heappop
from collections import defaultdict

def solve():
    N, Q = map(int, input().split())
    graph = defaultdict(list)
    heap = []
    for _ in range(Q):
        query = list(map(int, input().split()))
        if query[0] == 1:
            u, v = query[1:]
            graph[u].append(v)
            graph[v].append(u)
        else:
            v, k = query[1:]
            for u in graph[v]:
                heappush(heap, -u)
            ans = -1
            for _ in range(k):
                if heap:
                    ans = heappop(heap)
            if ans != -1:
                heappush(heap, ans)
            print(-ans if ans != -1 else -1)

solve()