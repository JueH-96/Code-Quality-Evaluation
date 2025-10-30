def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    index = 0
    N = int(data[index])
    index += 1
    K = int(data[index])
    index += 1
    
    from collections import defaultdict, deque
    
    tree = defaultdict(list)
    
    for _ in range(N - 1):
        A = int(data[index])
        index += 1
        B = int(data[index])
        index += 1
        tree[A].append(B)
        tree[B].append(A)
    
    specified_vertices = []
    for _ in range(K):
        V = int(data[index])
        index += 1
        specified_vertices.append(V)
    
    # We will use BFS to find the shortest path tree from any specified vertex
    # Let's choose the first specified vertex as the root
    root = specified_vertices[0]
    
    # BFS to find the parent and depth of each node
    parent = [-1] * (N + 1)
    depth = [-1] * (N + 1)
    visited = [False] * (N + 1)
    
    def bfs(root):
        queue = deque([root])
        depth[root] = 0
        visited[root] = True
        while queue:
            node = queue.popleft()
            for neighbor in tree[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    parent[neighbor] = node
                    depth[neighbor] = depth[node] + 1
                    queue.append(neighbor)
    
    bfs(root)
    
    # Function to find LCA of two nodes
    def lca(u, v):
        # Bring both nodes to the same depth
        if depth[u] < depth[v]:
            u, v = v, u
        # Lift u to the same depth as v
        while depth[u] > depth[v]:
            u = parent[u]
        # Now both are at the same depth, move both up until they meet
        while u != v:
            u = parent[u]
            v = parent[v]
        return u
    
    # Mark nodes that are part of the smallest subtree
    marked = [False] * (N + 1)
    
    # Mark all specified vertices
    for v in specified_vertices:
        marked[v] = True
    
    # For each pair of specified vertices, mark the path between them
    for i in range(K):
        for j in range(i + 1, K):
            u = specified_vertices[i]
            v = specified_vertices[j]
            lca_uv = lca(u, v)
            
            # Mark path from u to lca_uv
            while u != lca_uv:
                marked[u] = True
                u = parent[u]
            
            # Mark path from v to lca_uv
            while v != lca_uv:
                marked[v] = True
                v = parent[v]
            
            # Mark the LCA itself
            marked[lca_uv] = True
    
    # Count marked nodes
    result = sum(marked)
    
    print(result)