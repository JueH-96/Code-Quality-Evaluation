# YOUR CODE HERE
import sys, sys
import sys
import sys
sys.setrecursionlimit(1 << 25)

def solve():
    import sys
    from collections import defaultdict
    input = sys.stdin.read
    data = input().split()
    
    idx = 0
    N, M, K = map(int, data[idx:idx+3])
    idx += 3
    
    edges = []
    for _ in range(M):
        u, v, w = map(int, data[idx:idx+3])
        edges.append((w, u, v))
        idx += 3
    
    A = list(map(int, data[idx:idx+K]))
    idx += K
    B = list(map(int, data[idx:idx+K]))
    idx += K
    
    # Krusky's algorithm
    parent = list(range(N+1))
    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u
    def union(u, v):
        u_root = find(u)
        v_root = find(v)
        if u_root == v_root:
            return False
        parent[v_root] = u_root
        return True
    
    edges.sort()
    tree = [[] for _ in range(N+1)]
    for w, u, v in edges:
        if union(u, v):
            tree[u].append((v, w))
            tree[v].append((u, w))
    
    # Root the tree at node 1
    root = 1
    count_A = [0]*(N+1)
    count_B = [0]*(N+1)
    for a in A:
        count_A[a] += 1
    for b in B:
        count_B[b] += 1
    
    total_sum = 0
    def dfs(u, parent_node, edge_w):
        nonlocal total_sum
        total_a = count_A[u]
        total_b = count_B[u]
        for v, w in tree[u]:
            if v != parent_node:
                a, b = dfs(v, u, w)
                total_a += a
                total_b += b
        if parent_node != -1:
            matches = min(total_a, total_b)
            total_sum += matches * edge_w
            return (total_a - matches, total_b - matches)
        else:
            # Root node
            return (0, 0)
    
    dfs(root, -1, 0)
    print(total_sum)