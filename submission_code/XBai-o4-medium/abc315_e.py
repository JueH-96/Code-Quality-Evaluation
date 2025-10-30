import sys
from collections import deque, defaultdict

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1

    prereq = [[] for _ in range(N + 1)]  # 1-based indexing

    for i in range(1, N + 1):
        C_i = int(input[ptr])
        ptr += 1
        for _ in range(C_i):
            p = int(input[ptr])
            ptr += 1
            prereq[i].append(p)

    # Collect all required books using BFS starting from book 1
    visited = set()
    q = deque()
    q.append(1)
    visited.add(1)
    while q:
        current = q.popleft()
        for p in prereq[current]:
            if p not in visited:
                visited.add(p)
                q.append(p)

    required = visited

    # Build adjacency list and in_degree for the required subgraph
    adj = defaultdict(list)
    in_degree = {node: 0 for node in required}

    for u in required:
        for v in prereq[u]:
            adj[v].append(u)
            in_degree[u] += 1

    # Kahn's algorithm for topological sort
    topo = []
    q = deque()
    for node in required:
        if in_degree[node] == 0:
            q.append(node)

    while q:
        u = q.popleft()
        topo.append(u)
        for v in adj[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                q.append(v)

    # Remove book 1 from the topological order
    result = [x for x in topo if x != 1]
    print(' '.join(map(str, result)))

if __name__ == "__main__":
    main()