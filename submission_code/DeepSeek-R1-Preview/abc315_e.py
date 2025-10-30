import sys
from collections import deque

def main():
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    idx += 1
    
    prereq = [[] for _ in range(N + 1)]
    for i in range(1, N + 1):
        C_i = int(data[idx])
        idx += 1
        prereq[i] = list(map(int, data[idx:idx + C_i]))
        idx += C_i
    
    # Build reverse adjacency list
    rev_adj = [[] for _ in range(N + 1)]
    for i in range(1, N + 1):
        for p in prereq[i]:
            rev_adj[i].append(p)
    
    # BFS to find all nodes that can reach 1 in the original graph
    visited = [False] * (N + 1)
    q = deque()
    q.append(1)
    visited[1] = True
    while q:
        u = q.popleft()
        for v in rev_adj[u]:
            if not visited[v]:
                visited[v] = True
                q.append(v)
    
    # Collect required books (excluding 1)
    S = []
    for i in range(1, N + 1):
        if visited[i] and i != 1:
            S.append(i)
    
    # Compute in-degree for each node in S
    in_degree = [0] * (N + 1)
    for v in S:
        for p in prereq[v]:
            if visited[p]:  # p is in S
                in_degree[v] += 1
    
    # Kahn's algorithm
    queue = deque()
    for v in S:
        if in_degree[v] == 0:
            queue.append(v)
    
    topo_order = []
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        # Traverse original adjacency (prerequisites)
        # Wait, adj is the original graph, which is stored as prereq?
        # No, we need to find all books that have u as a prerequisite, i.e., adj[u]
        # So, build adj:
    adj = [[] for _ in range(N + 1)]
    for i in range(1, N + 1):
        for p in prereq[i]:
            adj[p].append(i)
    # Now, proceed with Kahn's
    in_degree = [0] * (N + 1)
    for v in S:
        for p in prereq[v]:
            if visited[p]:
                in_degree[v] += 1
    queue = deque()
    for v in S:
        if in_degree[v] == 0:
            queue.append(v)
    topo_order = []
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in adj[u]:
            if visited[v] and v != 1:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)
    
    print(' '.join(map(str, topo_order)))

if __name__ == '__main__':
    main()