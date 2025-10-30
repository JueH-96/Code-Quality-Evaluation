import sys
from collections import deque

def get_farthest(start, adj):
    max_dist = 0
    farthest = start
    dist = {start: 0}
    q = deque([start])
    while q:
        u = q.popleft()
        for v, w in adj[u]:
            if v not in dist:
                dist[v] = dist[u] + w
                q.append(v)
                if dist[v] > max_dist:
                    max_dist = dist[v]
                    farthest = v
    return farthest, max_dist

n = int(sys.stdin.readline())
adj = [[] for _ in range(n+1)]
total = 0
for _ in range(n-1):
    a, b, c = map(int, sys.stdin.readline().split())
    adj[a].append((b, c))
    adj[b].append((a, c))
    total += c

u, _ = get_farthest(1, adj)
v, diameter = get_farthest(u, adj)
print(2 * total - diameter)