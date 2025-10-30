import sys
from collections import deque

def main():
    N = int(sys.stdin.readline())
    prereqs = [[] for _ in range(N + 1)]  # 1-based indexing

    for i in range(1, N + 1):
        parts = list(map(int, sys.stdin.readline().split()))
        prereqs[i] = parts[1:]

    # Collect all books required to read book 1 (excluding book 1)
    visited = [False] * (N + 1)
    q = deque()
    q.append(1)
    visited[1] = True
    required = set()

    while q:
        u = q.popleft()
        for v in prereqs[u]:
            if not visited[v]:
                visited[v] = True
                q.append(v)
                required.add(v)

    # Build reversed adjacency list for topological sort
    reversed_adj = [[] for _ in range(N + 1)]
    in_degree = [0] * (N + 1)

    for u in required:
        for v in prereqs[u]:
            reversed_adj[v].append(u)
            in_degree[u] += 1

    # Perform Kahn's algorithm for topological sort
    queue = deque()
    for u in required:
        if in_degree[u] == 0:
            queue.append(u)
    
    result = []
    while queue:
        u = queue.popleft()
        result.append(u)
        for v in reversed_adj[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
    
    print(' '.join(map(str, result)))

if __name__ == "__main__":
    main()