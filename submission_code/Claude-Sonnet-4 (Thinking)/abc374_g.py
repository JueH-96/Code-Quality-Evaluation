def solve():
    n = int(input())
    if n == 0:
        print(0)
        return
    
    edges = []
    for _ in range(n):
        s = input().strip()
        edges.append((s[0], s[1]))
    
    from collections import defaultdict
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
    
    all_vertices = set()
    for u, v in edges:
        all_vertices.add(u)
        all_vertices.add(v)
    
    uncovered = set(edges)
    trails = 0
    
    while uncovered:
        # Find the trail that covers the most uncovered edges
        best_trail = None
        best_coverage = 0
        
        for start in all_vertices:
            # DFS to find the best trail starting from start
            def dfs(v, trail, depth):
                nonlocal best_trail, best_coverage
                
                if depth > n:  # Limit depth to avoid infinite loops
                    return
                
                # Calculate coverage
                coverage_edges = set()
                for i in range(len(trail) - 1):
                    edge = (trail[i], trail[i+1])
                    coverage_edges.add(edge)
                
                coverage_count = len(coverage_edges & uncovered)
                if coverage_count > best_coverage:
                    best_coverage = coverage_count
                    best_trail = trail[:]
                
                for neighbor in graph[v]:
                    trail.append(neighbor)
                    dfs(neighbor, trail, depth + 1)
                    trail.pop()
            
            dfs(start, [start], 1)
        
        if best_trail and best_coverage > 0:
            trails += 1
            # Remove covered edges
            for i in range(len(best_trail) - 1):
                edge = (best_trail[i], best_trail[i+1])
                if edge in uncovered:
                    uncovered.remove(edge)
        else:
            break
    
    print(trails)

solve()