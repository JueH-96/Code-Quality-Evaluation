import sys
from collections import deque

def readints():
    return list(map(int, sys.stdin.readline().split()))

def bfs(start, adj, n):
    visited = [False] * (n + 1)
    distance = [0] * (n + 1)
    q = deque()
    q.append(start)
    visited[start] = True
    max_dist = 0
    far_node = start
    while q:
        u = q.popleft()
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                distance[v] = distance[u] + 1
                if distance[v] > max_dist:
                    max_dist = distance[v]
                    far_node = v
                q.append(v)
    return far_node, distance

def find_diameter_path(adj, n):
    u, _ = bfs(1, adj, n)
    v, _ = bfs(u, adj, n)
    parent = [0] * (n + 1)
    visited = [False] * (n + 1)
    q = deque()
    q.append(u)
    visited[u] = True
    while q:
        current = q.popleft()
        for neighbor in adj[current]:
            if not visited[neighbor]:
                visited[neighbor] = True
                parent[neighbor] = current
                q.append(neighbor)
    path = []
    current = v
    while current != u:
        path.append(current)
        current = parent[current]
    path.append(u)
    path.reverse()
    return path

def main():
    n = int(sys.stdin.readline())
    adj = [[] for _ in range(n+1)]
    for _ in range(n-1):
        a, b = map(int, sys.stdin.readline().split())
        adj[a].append(b)
        adj[b].append(a)
    
    path = find_diameter_path(adj, n)
    m = len(path)
    pairs = []
    i = 0
    j = m - 1
    while i < j:
        pairs.append((path[i], path[j]))
        i += 1
        j -= 1
    remaining = set(range(1, n+1))
    for x, y in pairs:
        remaining.discard(x)
        remaining.discard(y)
    if remaining:
        remaining_list = list(remaining)
        i = 0
        j = len(remaining_list) - 1
        while i < j:
            pairs.append((remaining_list[i], remaining_list[j]))
            i += 1
            j -= 1
    for x, y in pairs:
        print(x, y)

if __name__ == "__main__":
    main()