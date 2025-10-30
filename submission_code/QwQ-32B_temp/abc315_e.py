import sys
from collections import deque

def main():
    sys.setrecursionlimit(1 << 25)
    N = int(sys.stdin.readline())
    prereq = [[] for _ in range(N + 1)]  # 1-based indexing
    for i in range(1, N + 1):
        parts = list(map(int, sys.stdin.readline().split()))
        C = parts[0]
        P_list = parts[1:]
        prereq[i] = P_list
    
    # Build reversed adjacency list for BFS to find required nodes
    reversed_adj = [[] for _ in range(N + 1)]
    for i in range(1, N + 1):
        for p in prereq[i]:
            reversed_adj[i].append(p)
    
    # Find all nodes reachable from book 1 in reversed graph (including 1)
    required = set()
    q = deque()
    q.append(1)
    required.add(1)
    while q:
        u = q.popleft()
        for v in reversed_adj[u]:
            if v not in required:
                required.add(v)
                q.append(v)
    
    # Collect required nodes except book 1
    nodes = [x for x in required if x != 1]
    
    # Compute in_degree for each node in nodes (C_i is the number of prerequisites)
    in_degree = [0] * (N + 1)
    for u in nodes:
        in_degree[u] = len(prereq[u])
    
    # Build adjacency list for the subgraph (original edges between required nodes)
    adj_sub = [[] for _ in range(N + 1)]
    for u in nodes:
        for p in prereq[u]:
            adj_sub[p].append(u)
    
    # Kahn's algorithm for topological sort
    q = deque()
    for u in nodes:
        if in_degree[u] == 0:
            q.append(u)
    
    order = []
    while q:
        u = q.popleft()
        order.append(u)
        for v in adj_sub[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                q.append(v)
    
    print(' '.join(map(str, order)))

if __name__ == "__main__":
    main()