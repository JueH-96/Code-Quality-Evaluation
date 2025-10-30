def can_assign_values(N, M, A, B):
    # Create graph where edges represent pairs that must be different
    graph = [[] for _ in range(N+1)]
    for i in range(M):
        a, b = A[i], B[i]
        if a == b:  # If same index must be different, impossible
            return False
        graph[a].append(b)
        graph[b].append(a)
    
    # Try 2-coloring using DFS
    colors = [None] * (N+1)
    
    def dfs(v, color):
        colors[v] = color
        for u in graph[v]:
            if colors[u] is None:
                if not dfs(u, 1-color):
                    return False
            elif colors[u] == color:
                return False
        return True
    
    # Try coloring each connected component
    for i in range(1, N+1):
        if colors[i] is None:
            if graph[i]:  # Only try coloring if vertex has edges
                if not dfs(i, 0):
                    return False
    
    return True

# Read input
N, M = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))

# Print result
print("Yes" if can_assign_values(N, M, A, B) else "No")