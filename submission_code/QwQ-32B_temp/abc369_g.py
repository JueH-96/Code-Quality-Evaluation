import sys
from collections import deque

def main():
    sys.setrecursionlimit(1 << 25)
    N = int(sys.stdin.readline())
    edges = [[] for _ in range(N+1)]  # 1-based
    for _ in range(N-1):
        u, v, l = map(int, sys.stdin.readline().split())
        edges[u].append( (v, l) )
        edges[v].append( (u, l) )
    
    # Compute degrees
    degree = [0] * (N+1)
    for u in range(1, N+1):
        degree[u] = len(edges[u])
    
    # BFS to find parent and length arrays
    parent = [0] * (N+1)
    length = [0] * (N+1)
    visited = [False] * (N+1)
    q = deque()
    q.append(1)
    visited[1] = True
    parent[1] = 0  # root has no parent
    
    while q:
        u = q.popleft()
        for v, l in edges[u]:
            if not visited[v] and v != parent[u]:
                parent[v] = u
                length[v] = l
                visited[v] = True
                q.append(v)
    
    # Collect leaf edges
    leaf_edges = []
    for v in range(2, N+1):
        if degree[v] == 1:
            leaf_edges.append(length[v])
    
    # Check root case
    if degree[1] == 1:
        # Find the child of root (must be one)
        child = -1
        for v in range(2, N+1):
            if parent[v] == 1:
                child = v
                break
        if child != -1 and degree[child] == 1:
            leaf_edges.append(length[child])
    
    # Sort leaf edges and compute prefix sums
    leaf_edges.sort()
    L = len(leaf_edges)
    prefix = [0] * (L + 1)
    for i in range(L):
        prefix[i+1] = prefix[i] + leaf_edges[i]
    
    # Compute total_sum
    total_sum = sum(length[v] for v in range(2, N+1))
    
    # Process each K from 1 to N
    for K in range(1, N+1):
        if K >= L:
            ans = total_sum
        else:
            ans = total_sum - prefix[L - K]
        print(2 * ans)
    
if __name__ == "__main__":
    main()