from collections import deque

def solve():
    N, M = map(int, input().split())
    
    # Build adjacency list
    adj = [[] for _ in range(N + 1)]
    for _ in range(M):
        u, v = map(int, input().split())
        adj[u].append(v)
        adj[v].append(u)
    
    # Color vertices using BFS (bipartite check)
    color = [-1] * (N + 1)
    component_sizes = []  # Store (size1, size2) for each component
    
    for start in range(1, N + 1):
        if color[start] == -1:
            # BFS to color this component
            queue = deque([start])
            color[start] = 0
            count = [0, 0]  # count[0] = vertices with color 0, count[1] = vertices with color 1
            
            while queue:
                u = queue.popleft()
                count[color[u]] += 1
                
                for v in adj[u]:
                    if color[v] == -1:
                        color[v] = 1 - color[u]
                        queue.append(v)
            
            component_sizes.append((count[0], count[1]))
    
    # Calculate total possible edges in complete bipartite graph
    total_possible_edges = 0
    for c0, c1 in component_sizes:
        total_possible_edges += c0 * c1
    
    # Number of moves remaining
    moves_remaining = total_possible_edges - M
    
    # If odd number of moves, first player wins
    if moves_remaining % 2 == 1:
        print("Aoki")
    else:
        print("Takahashi")

solve()