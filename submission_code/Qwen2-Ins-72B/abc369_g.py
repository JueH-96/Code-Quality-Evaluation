from heapq import heappush, heappop
from collections import deque

def solve(n, edges):
    tree = [[] for _ in range(n)]
    for u, v, l in edges:
        tree[u-1].append((v-1, l))
        tree[v-1].append((u-1, l))

    def dfs(v, parent, dist):
        stack = [(v, parent, dist)]
        while stack:
            v, parent, dist = stack.pop()
            for u, l in tree[v]:
                if u == parent:
                    continue
                dist_to_u = dist + l
                stack.append((u, v, dist_to_u))
                yield u, dist_to_u

    def find_furthest(v):
        furthest, _ = max(dfs(v, -1, 0), key=lambda x: x[1])
        furthest, _ = max(dfs(furthest, -1, 0), key=lambda x: x[1])
        return furthest

    diameter_end = find_furthest(0)
    _, diameter = max(dfs(diameter_end, -1, 0), key=lambda x: x[1])

    def dijkstra(start):
        dist = [float('inf')] * n
        dist[start] = 0
        pq = [(0, start)]
        while pq:
            d, v = heappop(pq)
            if d > dist[v]:
                continue
            for u, l in tree[v]:
                if dist[u] > dist[v] + l:
                    dist[u] = dist[v] + l
                    heappush(pq, (dist[u], u))
        return dist

    dist_from_0 = dijkstra(0)
    dist_from_diameter_end = dijkstra(diameter_end)

    def find_second_furthest(v):
        dist = dist_from_diameter_end[:]
        dist[v] = float('-inf')
        second_furthest = max(range(n), key=lambda x: dist[x])
        return second_furthest

    second_furthest = find_second_furthest(diameter_end)
    second_diameter = dist_from_diameter_end[second_furthest]

    def find_third_furthest(v):
        dist = dist_from_diameter_end[:]
        dist[v] = float('-inf')
        dist[second_furthest] = float('-inf')
        third_furthest = max(range(n), key=lambda x: dist[x])
        return third_furthest

    third_furthest = find_third_furthest(diameter_end)

    ans = [0] * n
    ans[0] = dist_from_0[diameter_end] * 2
    ans[1] = max(ans[0], dist_from_0[second_furthest] + dist_from_diameter_end[second_furthest])
    for k in range(2, n):
        ans[k] = max(ans[k-1], diameter + second_diameter + dist_from_0[third_furthest])

    return ans

n = int(input())
edges = [list(map(int, input().split())) for _ in range(n-1)]
for x in solve(n, edges):
    print(x)