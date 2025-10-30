def floyd_warshall(n, edges):
    # Initialize distance matrix with infinity
    dist = [[float('inf')] * n for _ in range(n)]
    
    # Set diagonal to 0
    for i in range(n):
        dist[i][i] = 0
        
    # Set initial edges
    for u, v, w in edges:
        dist[u-1][v-1] = w
        dist[v-1][u-1] = w
    
    # Floyd-Warshall algorithm
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] != float('inf') and dist[k][j] != float('inf'):
                    dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
    
    return dist

def main():
    # Read input
    N, M, Q = map(int, input().split())
    
    # Store edges
    edges = []
    for _ in range(M):
        a, b, c = map(int, input().split())
        edges.append((a, b, c))
    
    # Store which edges are active (1) or closed (0)
    active_edges = [1] * M
    
    # Process queries
    for _ in range(Q):
        query = list(map(int, input().split()))
        
        if query[0] == 1:
            # Close road i
            i = query[1]
            active_edges[i-1] = 0
            
            # Recalculate shortest paths with current active edges
            current_edges = [(edges[i][0], edges[i][1], edges[i][2]) 
                           for i in range(M) if active_edges[i]]
            distances = floyd_warshall(N, current_edges)
            
        else:
            # Query shortest path from x to y
            x, y = query[1], query[2]
            dist = distances[x-1][y-1]
            print(-1 if dist == float('inf') else dist)

if __name__ == "__main__":
    main()