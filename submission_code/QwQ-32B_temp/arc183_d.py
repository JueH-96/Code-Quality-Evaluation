import sys
from collections import deque

def main():
    sys.setrecursionlimit(1 << 25)
    N = int(sys.stdin.readline())
    adj = [[] for _ in range(N+1)]
    degree = [0]*(N+1)
    for _ in range(N-1):
        a, b = map(int, sys.stdin.readline().split())
        adj[a].append(b)
        adj[b].append(a)
        degree[a] += 1
        degree[b] += 1

    def bfs(start):
        visited = [False]*(N+1)
        parent = [0]*(N+1)
        distance = [0]*(N+1)
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
                    parent[v] = u
                    distance[v] = distance[u] + 1
                    if distance[v] > max_dist:
                        max_dist = distance[v]
                        far_node = v
                    q.append(v)
        return far_node, parent, distance

    u_start = 1
    u, _, _ = bfs(u_start)
    v, parent_v, _ = bfs(u)

    path = []
    current = v
    while current != u:
        path.append(current)
        current = parent_v[current]
    path.append(u)
    path = path[::-1]

    mid = len(path) // 2
    mid_node = path[mid]

    distance_from_mid = [0]*(N+1)
    visited = [False]*(N+1)
    q = deque()
    q.append(mid_node)
    visited[mid_node] = True
    while q:
        u = q.popleft()
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                distance_from_mid[v] = distance_from_mid[u] + 1
                q.append(v)

    leaves = [node for node in range(1, N+1) if degree[node] == 1]
    leaves.sort(key=lambda x: -distance_from_mid[x])

    result = []
    for i in range(0, len(leaves), 2):
        a = leaves[i]
        b = leaves[i+1]
        result.append((a, b))

    for pair in result:
        print(pair[0], pair[1])

if __name__ == "__main__":
    main()