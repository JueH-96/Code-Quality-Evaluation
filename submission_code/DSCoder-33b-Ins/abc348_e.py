import sys
from collections import defaultdict
from heapq import heappop, heappush

def solve():
    n = int(input())
    graph = defaultdict(list)
    for _ in range(n - 1):
        a, b = map(int, input().split())
        graph[a].append(b)
        graph[b].append(a)
    c = list(map(int, input().split()))
    dist = [0] * (n + 1)
    parent = [0] * (n + 1)
    queue = [(0, 1, 0)]
    while queue:
        d, node, p = heappop(queue)
        if dist[node] > 0:
            continue
        dist[node] = d
        parent[node] = p
        for nei in graph[node]:
            if dist[nei] == 0:
                heappush(queue, (d + 1, nei, node))
    ans = float('inf')
    for i in range(1, n + 1):
        ans = min(ans, sum(dist[j] * c[j - 1] for j in range(1, n + 1)))
    print(ans)

solve()