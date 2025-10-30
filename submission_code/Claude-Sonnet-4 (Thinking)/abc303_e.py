def solve():
    n = int(input())
    edges = []
    for _ in range(n - 1):
        u, v = map(int, input().split())
        edges.append((u, v))
    
    # Build adjacency list
    adj = [[] for _ in range(n + 1)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    stars = []
    
    # Find vertices with degree >= 3 (definitely star centers)
    high_degree_vertices = 0
    high_degree_sum = 0
    
    for i in range(1, n + 1):
        if len(adj[i]) >= 3:
            stars.append(len(adj[i]))
            high_degree_vertices += 1
            high_degree_sum += len(adj[i])
    
    # Calculate remaining vertices and estimate level-2 stars
    remaining_vertices = n - high_degree_sum - high_degree_vertices
    
    # Each level-2 star contributes 2 additional vertices beyond its center
    # So number of level-2 stars = remaining_vertices / 2
    level2_stars = remaining_vertices // 2
    
    # Add level-2 stars to result
    stars.extend([2] * level2_stars)
    
    stars.sort()
    print(' '.join(map(str, stars)))

solve()