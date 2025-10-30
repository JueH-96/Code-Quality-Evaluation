def floyd_warshall(n, edges):
    # Initialize distance matrix
    dist = [[float('inf')] * (n + 1) for _ in range(n + 1)]
    
    # Distance from a node to itself is 0
    for i in range(1, n + 1):
        dist[i][i] = 0
    
    # Add all edges
    for u, v, t in edges:
        dist[u][v] = min(dist[u][v], t)
        dist[v][u] = min(dist[v][u], t)
    
    # Floyd-Warshall algorithm
    for k in range(1, n + 1):
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
    
    return dist

def solve_query(n, edges, dist, required_bridges):
    from itertools import permutations
    
    min_cost = float('inf')
    
    # Try all permutations of required bridges
    for perm in permutations(range(len(required_bridges))):
        # Try both directions for each bridge
        for directions in range(1 << len(required_bridges)):
            cost = 0
            current = 1  # Start at island 1
            
            valid = True
            for i, bridge_idx in enumerate(perm):
                bridge = required_bridges[bridge_idx]
                u, v, t = edges[bridge - 1]
                
                # Determine direction based on bit
                if (directions >> i) & 1:
                    # Go from current to v, then traverse bridge from v to u
                    cost += dist[current][v] + t
                    current = u
                else:
                    # Go from current to u, then traverse bridge from u to v
                    cost += dist[current][u] + t
                    current = v
            
            # Add cost to reach island N from current position
            cost += dist[current][n]
            
            min_cost = min(min_cost, cost)
    
    return min_cost

# Read input
n, m = map(int, input().split())
edges = []
for _ in range(m):
    u, v, t = map(int, input().split())
    edges.append((u, v, t))

# Compute all-pairs shortest paths
dist = floyd_warshall(n, edges)

# Process queries
q = int(input())
for _ in range(q):
    k = int(input())
    required_bridges = list(map(int, input().split()))
    
    result = solve_query(n, edges, dist, required_bridges)
    print(result)