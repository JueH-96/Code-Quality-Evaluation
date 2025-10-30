def find_minimum_vertices(N, K, edges, vertices):
    from collections import defaultdict, deque
    
    # Build the adjacency list for the tree
    tree = defaultdict(list)
    for a, b in edges:
        tree[a].append(b)
        tree[b].append(a)
    
    # Function to find the LCA (Lowest Common Ancestor) using DFS
    def dfs(node, parent):
        depth[node] = depth[parent] + 1 if parent != -1 else 0
        ancestors[node][0] = parent
        for i in range(1, max_log):
            if ancestors[node][i - 1] != -1:
                ancestors[node][i] = ancestors[ancestors[node][i - 1]][i - 1]
        
        for neighbor in tree[node]:
            if neighbor != parent:
                dfs(neighbor, node)
    
    def lca(u, v):
        if depth[u] < depth[v]:
            u, v = v, u
        
        # Bring u and v to the same depth
        for i in range(max_log - 1, -1, -1):
            if depth[ancestors[u][i]] >= depth[v]:
                u = ancestors[u][i]
        
        if u == v:
            return u
        
        for i in range(max_log - 1, -1, -1):
            if ancestors[u][i] != ancestors[v][i]:
                u = ancestors[u][i]
                v = ancestors[v][i]
        
        return ancestors[u][0]
    
    # Prepare for LCA
    max_log = 18  # Since N <= 200000, log2(200000) < 18
    depth = [-1] * (N + 1)
    ancestors = [[-1] * max_log for _ in range(N + 1)]
    
    # Start DFS from node 1 (or any node, since it's a tree)
    dfs(1, -1)
    
    # Find the LCA of all specified vertices
    current_lca = vertices[0]
    for v in vertices[1:]:
        current_lca = lca(current_lca, v)
    
    # Count the number of vertices in the subtree rooted at the LCA
    def count_subtree_size(node, parent):
        size = 1
        for neighbor in tree[node]:
            if neighbor != parent:
                size += count_subtree_size(neighbor, node)
        return size
    
    # The size of the subtree rooted at the LCA
    subtree_size = count_subtree_size(current_lca, -1)
    
    return subtree_size

import sys
input = sys.stdin.read
data = input().splitlines()

# Read N and K
N, K = map(int, data[0].split())
edges = [tuple(map(int, line.split())) for line in data[1:N]]
vertices = list(map(int, data[N].split()))

# Get the result
result = find_minimum_vertices(N, K, edges, vertices)

# Print the result
print(result)