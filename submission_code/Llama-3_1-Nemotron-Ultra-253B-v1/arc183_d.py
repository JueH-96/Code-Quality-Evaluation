import sys
from collections import deque

def input():
    return sys.stdin.read()

def bfs(start, adj, n):
    dist = [-1] * (n + 1)
    q = deque()
    q.append(start)
    dist[start] = 0
    while q:
        u = q.popleft()
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                q.append(v)
    max_dist = max(dist[1:])
    far_node = dist.index(max_dist)
    return far_node, dist

def find_diameter(adj, n):
    u, _ = bfs(1, adj, n)
    v, dist_u = bfs(u, adj, n)
    _, dist_v = bfs(v, adj, n)
    diameter = []
    path = []
    parent = [0] * (n + 1)
    visited = [False] * (n + 1)
    q = deque()
    q.append(v)
    visited[v] = True
    while q:
        node = q.popleft()
        for neighbor in adj[node]:
            if not visited[neighbor]:
                visited[neighbor] = True
                parent[neighbor] = node
                q.append(neighbor)
    current = u
    while current != v:
        diameter.append(current)
        current = parent[current]
    diameter.append(v)
    return diameter

def main():
    data = input().split()
    idx = 0
    n = int(data[idx])
    idx +=1
    adj = [[] for _ in range(n+1)]
    for _ in range(n-1):
        a = int(data[idx])
        b = int(data[idx+1])
        adj[a].append(b)
        adj[b].append(a)
        idx +=2

    removed = [False] * (n + 1)
    result = []

    for _ in range(n // 2):
        current_adj = [[] for _ in range(n+1)]
        for u in range(1, n+1):
            if not removed[u]:
                for v in adj[u]:
                    if not removed[v]:
                        current_adj[u].append(v)
        current_nodes = [u for u in range(1, n+1) if not removed[u]]
        if not current_nodes:
            break
        start = current_nodes[0]
        u, _ = bfs(start, current_adj, n)
        v, _ = bfs(u, current_adj, n)
        path = []
        parent = [0]*(n+1)
        visited = [False]*(n+1)
        q = deque([v])
        visited[v] = True
        while q:
            node = q.popleft()
            for neighbor in current_adj[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    parent[neighbor] = node
                    q.append(neighbor)
        current = u
        while current != v:
            path.append(current)
            current = parent[current]
        path.append(v)
        a, b = path[0], path[-1]
        result.append((a, b))
        removed[a] = True
        removed[b] = True

    for a, b in result:
        print(a, b)

if __name__ == "__main__":
    main()