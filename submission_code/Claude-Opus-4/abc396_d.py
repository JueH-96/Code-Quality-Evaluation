# YOUR CODE HERE
def solve():
    N, M = map(int, input().split())
    
    # Build adjacency list
    graph = [[] for _ in range(N + 1)]
    
    for _ in range(M):
        u, v, w = map(int, input().split())
        graph[u].append((v, w))
        graph[v].append((u, w))
    
    # Find minimum XOR using DFS
    min_xor = float('inf')
    
    def dfs(node, target, visited, current_xor):
        nonlocal min_xor
        
        if node == target:
            min_xor = min(min_xor, current_xor)
            return
        
        visited.add(node)
        
        for neighbor, weight in graph[node]:
            if neighbor not in visited:
                dfs(neighbor, target, visited, current_xor ^ weight)
        
        visited.remove(node)
    
    # Start DFS from vertex 1 to vertex N
    dfs(1, N, set(), 0)
    
    print(min_xor)

solve()