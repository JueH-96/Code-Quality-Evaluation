def solve():
    N = int(input())
    S = input()
    T = input()
    
    if S == T:
        return 0
    
    # Check if the mapping is consistent
    char_map = {}
    for i in range(N):
        if S[i] not in char_map:
            char_map[S[i]] = T[i]
        elif char_map[S[i]] != T[i]:
            return -1  # Conflict in mapping: not possible
    
    # Build a directed graph from the mappings
    graph = {}
    for char_s, char_t in char_map.items():
        if char_s != char_t:  # Only consider mappings where s != t
            graph[char_s] = char_t
    
    # Count the unique transformations needed
    edges = len(graph)
    
    # Check for cycles in the graph
    cycles = 0
    visited = set()
    path = set()
    
    def dfs(node):
        nonlocal cycles
        if node in path:
            cycles += 1  # Found a cycle
            return
        if node in visited:
            return
        
        visited.add(node)
        path.add(node)
        
        if node in graph:
            dfs(graph[node])
        
        path.remove(node)
    
    for node in graph:
        if node not in visited:
            dfs(node)
    
    return edges + cycles  # Total operations = transformations + cycles

# Read input, solve the problem, and print the result
print(solve())