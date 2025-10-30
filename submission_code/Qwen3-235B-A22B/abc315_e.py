import sys
from collections import deque

def main():
    input = sys.stdin.read().split()
    ptr = 0

    N = int(input[ptr])
    ptr += 1

    prerequisites = [[] for _ in range(N + 1)]  # 1-based indexing

    for i in range(1, N + 1):
        C_i = int(input[ptr])
        ptr += 1
        for j in range(C_i):
            p = int(input[ptr])
            ptr += 1
            prerequisites[i].append(p)

    # Step 1: collect required nodes using BFS
    required = set()
    q = deque()

    for p in prerequisites[1]:
        if p not in required:
            required.add(p)
            q.append(p)

    while q:
        u = q.popleft()
        for p in prerequisites[u]:
            if p not in required:
                required.add(p)
                q.append(p)

    # Step 2: build adjacency list and in_degree
    adj = [[] for _ in range(N + 1)]
    in_degree = [0] * (N + 1)

    for node in required:
        for p in prerequisites[node]:
            adj[p].append(node)
            in_degree[node] += 1

    # Step 3: Kahn's algorithm for topological sort
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

    # Output the result
    print(' '.join(map(str, order)))

if __name__ == '__main__':
    main()