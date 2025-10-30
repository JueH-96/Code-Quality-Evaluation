def solve():
    n = int(input())
    graph = [[] for _ in range(n + 1)]
    
    for _ in range(n - 1):
        u, v = map(int, input().split())
        graph[u].append(v)
        graph[v].append(u)
    
    degrees = [len(graph[i]) for i in range(n + 1)]
    
    # Key insight: vertices with degree >= 3 are likely original star centers
    # The degree of such a vertex in the final tree equals its original star level
    # plus the number of times its component was connected to other components
    
    # But we need to be more careful about the reconstruction
    
    # Alternative approach: use the constraint that the sum of star levels
    # equals n (since each star of level k contributes k+1 vertices)
    
    # Find all vertices with degree > 2
    high_degree_vertices = []
    for i in range(1, n + 1):
        if degrees[i] > 2:
            high_degree_vertices.append((degrees[i], i))
    
    high_degree_vertices.sort()
    
    # The degrees of these vertices give us information about star levels
    star_levels = []
    
    if not high_degree_vertices:
        # All vertices have degree <= 2, so we have a path
        # This comes from connecting level-2 stars
        num_stars = (n + 1) // 3
        star_levels = [2] * num_stars
    else:
        # We have some high-degree vertices
        # Each represents a star center
        for degree, vertex in high_degree_vertices:
            star_levels.append(degree)
        
        # We might need additional level-2 stars
        total_vertices_accounted = sum(level + 1 for level in star_levels)
        remaining = n - total_vertices_accounted
        
        # Add level-2 stars for remaining vertices
        while remaining > 0:
            star_levels.append(2)
            remaining -= 3
    
    star_levels.sort()
    print(" ".join(map(str, star_levels)))

solve()