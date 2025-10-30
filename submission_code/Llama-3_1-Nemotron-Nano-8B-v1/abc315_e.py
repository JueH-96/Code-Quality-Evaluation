import sys
from collections import deque

def main():
    n = int(sys.stdin.readline())
    dependencies = [[] for _ in range(n + 1)]  # 1-based indexing

    for i in range(1, n + 1):
        parts = list(map(int, sys.stdin.readline().split()))
        c_i = parts[0]
        ps = parts[1:]
        dependencies[i] = ps

    # Collect all required books except book 1
    required = set()
    visited = [False] * (n + 1)
    q = deque()

    # Start with dependencies of book 1
    for p in dependencies[1]:
        if not visited[p]:
            visited[p] = True
            q.append(p)
            required.add(p)

    while q:
        current = q.popleft()
        for p in dependencies[current]:
            if not visited[p]:
                visited[p] = True
                q.append(p)
                required.add(p)

    # Build adjacency list and in-degree for the required nodes
    adj = [[] for _ in range(n + 1)]
    in_degree = [0] * (n + 1)

    for node in required:
        for dep in dependencies[node]:
            adj[dep].append(node)
            in_degree[node] += 1

    # Kahn's algorithm for topological sort
    q = deque()
    for node in required:
        if in_degree[node] == 0:
            q.append(node)

    order = []
    while q:
        u = q.popleft()
        order.append(u)
        for v in adj[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                q.append(v)

    print(' '.join(map(str, order)))

if __name__ == "__main__":
    main()