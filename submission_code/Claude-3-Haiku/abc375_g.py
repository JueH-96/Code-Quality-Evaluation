import sys
from collections import deque

def bfs(n, m, roads, exclude_road):
    dist = [float('inf')] * (n + 1)
    dist[1] = 0
    queue = deque([1])
    while queue:
        u = queue.popleft()
        for v, c in roads[u]:
            if v != exclude_road and dist[v] > dist[u] + c:
                dist[v] = dist[u] + c
                queue.append(v)
    return dist[n] if dist[n] != float('inf') else -1

def solve():
    n, m = map(int, input().split())
    roads = [[] for _ in range(n + 1)]
    for _ in range(m):
        a, b, c = map(int, input().split())
        roads[a].append((b, c))
        roads[b].append((a, c))

    for i in range(m):
        dist1 = bfs(n, m, roads, i + 1)
        dist2 = bfs(n, m, roads, 0)
        print("Yes" if dist1 != dist2 else "No")

if __name__ == "__main__":
    solve()