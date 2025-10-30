def has_odd_cycle(graph, n):
    # Check if adding edge (u,v) creates an odd cycle using DFS
    def dfs(u, v, color, parent):
        if color[u] != -1:
            # If we reach a colored vertex, check if cycle length is odd
            path_len = color[v] - color[u] + 1
            return path_len % 2 == 1
        
        color[u] = color[v] + 1
        for next_v in graph[u]:
            if next_v != parent:
                if dfs(next_v, u, color, u):
                    return True
        return False

    color = [-1] * (n + 1)
    color[v] = 0
    return dfs(u, v, color, -1)

def can_add_edge(u, v, graph, n, edges):
    if (u, v) in edges or (v, u) in edges:
        return False
        
    # Temporarily add edge
    graph[u].add(v)
    graph[v].add(u)
    
    # Check for odd cycle
    has_cycle = has_odd_cycle(graph, n)
    
    # Remove temporary edge
    graph[u].remove(v)
    graph[v].remove(u)
    
    return not has_cycle

def solve():
    # Read input
    n, m = map(int, input().split())
    
    # Create adjacency list representation
    graph = [set() for _ in range(n + 1)]
    edges = set()
    
    # Add initial edges
    for _ in range(m):
        u, v = map(int, input().split())
        graph[u].add(v)
        graph[v].add(u)
        edges.add((u, v))
        edges.add((v, u))
    
    # Count possible moves
    possible_moves = 0
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            if can_add_edge(i, j, graph, n, edges):
                possible_moves += 1
    
    # If possible moves is odd, Aoki wins
    # If possible moves is even, Takahashi wins
    if possible_moves % 2 == 1:
        print("Aoki")
    else:
        print("Takahashi")

solve()