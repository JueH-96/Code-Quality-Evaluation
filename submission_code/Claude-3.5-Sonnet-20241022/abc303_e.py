from collections import defaultdict

def get_degree(adj, n):
    degree = [0] * (n+1)
    for u in adj:
        degree[u] = len(adj[u])
    return degree

def find_stars(n, edges):
    # Build adjacency list
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Get initial degrees
    degree = get_degree(adj, n)
    
    # Find centers (vertices with degree > 1)
    centers = []
    for v in range(1, n+1):
        if degree[v] > 1:
            centers.append(v)
    
    # For each center, count leaves connected to it
    star_levels = []
    visited = set()
    
    for center in centers:
        if center in visited:
            continue
            
        # Count direct leaves
        leaves = 0
        for v in adj[center]:
            if degree[v] == 1:
                leaves += 1
                visited.add(v)
        
        if leaves > 0:
            star_levels.append(leaves)
            visited.add(center)
    
    return sorted(star_levels)

def main():
    # Read input
    n = int(input())
    edges = []
    for _ in range(n-1):
        u, v = map(int, input().split())
        edges.append((u, v))
    
    # Find stars
    result = find_stars(n, edges)
    
    # Print result
    print(*result)

main()