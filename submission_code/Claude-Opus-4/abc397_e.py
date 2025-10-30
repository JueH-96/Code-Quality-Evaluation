def solve():
    N, K = map(int, input().split())
    
    if K == 1:
        # Special case: paths of length 1 (single vertices)
        # Always possible for a tree
        print("Yes")
        return
    
    # Build adjacency list
    adj = [[] for _ in range(N * K + 1)]
    for _ in range(N * K - 1):
        u, v = map(int, input().split())
        adj[u].append(v)
        adj[v].append(u)
    
    # Count vertices by degree
    degree_count = [0] * (N * K + 1)
    for i in range(1, N * K + 1):
        degree_count[len(adj[i])] += 1
    
    # For a valid decomposition into paths of length K:
    # - We need exactly 2N endpoints (2 per path)
    # - Leaves (degree 1) must be endpoints
    # - We need to check if we can form N paths
    
    leaves = degree_count[1]
    
    # If we have more than 2N leaves, impossible
    if leaves > 2 * N:
        print("No")
        return
    
    # Try to find a valid decomposition using DFS
    visited = [False] * (N * K + 1)
    paths_found = 0
    
    def dfs_path(v, parent, path):
        path.append(v)
        visited[v] = True
        
        if len(path) == K:
            return True
        
        for u in adj[v]:
            if u != parent and not visited[u]:
                if dfs_path(u, v, path):
                    return True
        
        path.pop()
        visited[v] = False
        return False
    
    # Start from leaves
    for i in range(1, N * K + 1):
        if not visited[i] and len(adj[i]) == 1:
            path = []
            if dfs_path(i, -1, path):
                paths_found += 1
                if paths_found == N:
                    print("Yes")
                    return
            else:
                # Reset visited for this failed attempt
                for v in path:
                    visited[v] = False
    
    # If not enough paths found starting from leaves, try other vertices
    for i in range(1, N * K + 1):
        if not visited[i]:
            path = []
            if dfs_path(i, -1, path):
                paths_found += 1
                if paths_found == N:
                    print("Yes")
                    return
            else:
                for v in path:
                    visited[v] = False
    
    print("No")

solve()