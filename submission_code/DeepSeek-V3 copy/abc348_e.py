import sys
from collections import deque

def main():
    sys.setrecursionlimit(1 << 25)
    N = int(sys.stdin.readline())
    edges = [[] for _ in range(N+1)]
    for _ in range(N-1):
        A, B = map(int, sys.stdin.readline().split())
        edges[A].append(B)
        edges[B].append(A)
    C = list(map(int, sys.stdin.readline().split()))
    
    # Compute the sum of C_i for all nodes
    total_C = sum(C)
    
    # Compute the sum of C_i * depth_i for all nodes, using node 1 as root
    depth = [0] * (N+1)
    parent = [0] * (N+1)
    q = deque()
    q.append(1)
    while q:
        u = q.popleft()
        for v in edges[u]:
            if v != parent[u]:
                parent[v] = u
                depth[v] = depth[u] + 1
                q.append(v)
    
    sum_C_depth = sum(C[i-1] * depth[i] for i in range(1, N+1))
    
    # Compute f(1)
    f1 = sum_C_depth
    
    # Now, we need to compute f(v) for all v, using the fact that f(v) = f(u) + (total_C - 2 * subtree_C[u]) * 1
    # where u is the parent of v, and subtree_C[u] is the sum of C_i in the subtree rooted at u
    
    # First, compute the subtree_C for all nodes
    subtree_C = [0] * (N+1)
    def dfs(u):
        subtree_C[u] = C[u-1]
        for v in edges[u]:
            if v != parent[u]:
                dfs(v)
                subtree_C[u] += subtree_C[v]
    dfs(1)
    
    # Now, compute f(v) for all v
    f = [0] * (N+1)
    f[1] = f1
    q = deque()
    q.append(1)
    while q:
        u = q.popleft()
        for v in edges[u]:
            if v != parent[u]:
                f[v] = f[u] + (total_C - 2 * subtree_C[v])
                q.append(v)
    
    # Find the minimum f(v)
    min_f = min(f[1:])
    print(min_f)

if __name__ == "__main__":
    main()