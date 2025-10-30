def find_min_xor_path(N, edges):
    # Construct the graph as an adjacency list
    graph = [[] for _ in range(N + 1)]
    for u, v, w in edges:
        graph[u].append((v, w))
        graph[v].append((u, w))  # Undirected graph
    
    # Variable to store the minimum XOR
    min_xor = float('inf')
    
    # DFS to generate all simple paths
    def dfs(node, visited, xor_so_far):
        nonlocal min_xor
        
        # If we reached the destination
        if node == N:
            min_xor = min(min_xor, xor_so_far)
            return
        
        visited[node] = True
        
        for neighbor, weight in graph[node]:
            if not visited[neighbor]:
                dfs(neighbor, visited, xor_so_far ^ weight)
        
        visited[node] = False  # Backtrack
    
    visited = [False] * (N + 1)
    dfs(1, visited, 0)
    
    return min_xor

# Read input
N, M = map(int, input().split())
edges = []
for _ in range(M):
    u, v, w = map(int, input().split())
    edges.append((u, v, w))

# Find the minimum XOR path
result = find_min_xor_path(N, edges)
print(result)