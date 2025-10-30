from collections import defaultdict, deque

def solve():
    N, M = map(int, input().split())
    
    # Build adjacency list
    graph = defaultdict(list)
    for _ in range(M):
        u, v = map(int, input().split())
        graph[u].append(v)
        graph[v].append(u)
    
    # Find connected components and check bipartiteness
    visited = [False] * (N + 1)
    color = [-1] * (N + 1)
    
    total_max_edges = 0
    
    for start in range(1, N + 1):
        if visited[start]:
            continue
            
        # BFS to find connected component and check bipartiteness
        queue = deque([start])
        visited[start] = True
        color[start] = 0
        component = [start]
        
        while queue:
            node = queue.popleft()
            for neighbor in graph[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    color[neighbor] = 1 - color[node]
                    component.append(neighbor)
                    queue.append(neighbor)
        
        # Count vertices in each partition of this component
        partition_0 = sum(1 for v in component if color[v] == 0)
        partition_1 = len(component) - partition_0
        
        # Maximum edges in this component
        total_max_edges += partition_0 * partition_1
    
    # Number of edges we can still add
    remaining_moves = total_max_edges - M
    
    # If remaining moves is odd, first player (Aoki) wins
    if remaining_moves % 2 == 1:
        print("Aoki")
    else:
        print("Takahashi")

solve()