import heapq
import sys

# It's good practice to encapsulate the main logic in a function.
def solve():
    # Read N and M (number of vertices and edges)
    N, M = map(int, sys.stdin.readline().split())
    
    # Read vertex weights A_i.
    # We store them in a 0-indexed list, so A[i] corresponds to A_{i+1} in problem statement.
    A = list(map(int, sys.stdin.readline().split()))
    
    # Initialize adjacency list for the graph.
    # adj[u] will store a list of (neighbor_v, edge_weight_B_uv) tuples.
    # Vertices are 0-indexed internally (0 to N-1).
    adj = [[] for _ in range(N)]
    
    # Read M edges and populate the adjacency list.
    for _ in range(M):
        u, v, b = map(int, sys.stdin.readline().split())
        # Convert vertex IDs from 1-indexed (problem statement) to 0-indexed (code).
        u -= 1
        v -= 1
        # Add edges for an undirected graph.
        adj[u].append((v, b))
        adj[v].append((u, b))
        
    # Initialize distance array. dist[i] will store the minimum path weight
    # from vertex 1 (0-indexed: vertex 0) to vertex i.
    # Set all distances to infinity initially.
    dist = [float('inf')] * N
    
    # Priority queue for Dijkstra's algorithm.
    # Stores tuples of (current_path_weight, vertex_id).
    # heapq is a min-heap, so it will always pop the smallest path_weight.
    pq = []
    
    # The starting vertex is 1 (which is 0-indexed vertex 0).
    # The weight of a path is defined as sum of vertex and edge weights appearing on it.
    # For the start vertex itself, the path weight is just its vertex weight.
    dist[0] = A[0]
    # Push the starting vertex into the priority queue.
    heapq.heappush(pq, (dist[0], 0))
    
    # Dijkstra's main loop
    while pq:
        # Pop the vertex with the smallest current path weight.
        d_u, u = heapq.heappop(pq)
        
        # If we have already found a shorter path to 'u' than 'd_u',
        # this means 'd_u' is an outdated entry in the priority queue. Skip it.
        # This check is crucial for performance and correctness in certain cases.
        if d_u > dist[u]:
            continue
            
        # Explore all neighbors 'v' of the current vertex 'u'.
        for v, b_uv in adj[u]:
            # Calculate the path weight to 'v' through 'u'.
            # It's (path weight to u) + (edge weight u-v) + (vertex weight of v).
            new_dist_v = d_u + b_uv + A[v]
            
            # If this new path to 'v' is shorter than the currently known shortest path to 'v',
            # update dist[v] and add 'v' to the priority queue.
            if new_dist_v < dist[v]:
                dist[v] = new_dist_v
                heapq.heappush(pq, (dist[v], v))
                
    # The problem asks for the minimum path weights for i = 2, 3, ..., N.
    # In 0-indexed terms, this corresponds to vertices 1, 2, ..., N-1.
    # Convert these distances to strings and join them with spaces for output.
    results = [str(dist[i]) for i in range(1, N)]
    print(" ".join(results))

# Call the solve function to run the program.
# This pattern is common in competitive programming to avoid global variables
# and ensure all logic is run when the script is executed.
if __name__ == "__main__":
    solve()