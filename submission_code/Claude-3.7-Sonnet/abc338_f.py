import sys
import math

def main():
    # Read input
    N, M = map(int, input().split())
    
    # Initialize graph
    graph = {}
    for i in range(1, N+1):
        graph[i] = []
    
    for _ in range(M):
        u, v, w = map(int, input().split())
        graph[u].append((v, w))
    
    # Compute shortest paths between all pairs of vertices
    shortest_paths = floyd_warshall(graph, N)
    
    # Check if there's a vertex from which all other vertices are reachable
    if not can_visit_all_vertices(shortest_paths, N):
        print("No")
        return
    
    # Compute the minimum weight walk
    min_weight = held_karp(shortest_paths, N)
    
    print(min_weight)

def can_visit_all_vertices(shortest_paths, N):
    """
    Check if there's a vertex from which all other vertices are reachable.
    """
    for i in range(1, N+1):
        can_reach_all = True
        for j in range(1, N+1):
            if shortest_paths[i][j] == math.inf:
                can_reach_all = False
                break
        if can_reach_all:
            return True
    return False

def floyd_warshall(graph, N):
    """
    Compute the shortest paths between all pairs of vertices.
    """
    # Initialize the distance matrix
    dist = [[math.inf for _ in range(N+1)] for _ in range(N+1)]
    
    # The distance from a vertex to itself is 0
    for i in range(1, N+1):
        dist[i][i] = 0
    
    # Set the distances for the edges in the graph
    for u in graph:
        for v, w in graph[u]:
            dist[u][v] = w
    
    # Apply the Floyd-Warshall algorithm
    for k in range(1, N+1):
        for i in range(1, N+1):
            for j in range(1, N+1):
                if dist[i][k] != math.inf and dist[k][j] != math.inf and dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
    
    return dist

def held_karp(shortest_paths, N):
    """
    Compute the minimum weight of a walk that visits every vertex.
    """
    # We'll use a dictionary to store the DP values
    dp = {}
    
    # Base cases
    for v in range(1, N+1):
        dp[(1 << (v-1), v)] = 0
    
    # Iterate over all possible bitmasks
    for mask in range(1, (1 << N)):
        for v in range(1, N+1):
            # If the bit for vertex v is not set in the mask, skip
            if not (mask & (1 << (v-1))):
                continue
            
            # If there's only one bit set in the mask (i.e., only vertex v is visited), we've already set the base case
            if mask == (1 << (v-1)):
                continue
            
            # Compute the minimum weight for the current state
            prev_mask = mask ^ (1 << (v-1))  # Remove vertex v from the mask
            dp[(mask, v)] = math.inf
            
            for u in range(1, N+1):
                # If the bit for vertex u is not set in the previous mask, or if there's no path from u to v, skip
                if not (prev_mask & (1 << (u-1))) or shortest_paths[u][v] == math.inf:
                    continue
                
                if (prev_mask, u) in dp:  # Ensure the previous state exists
                    dp[(mask, v)] = min(dp[(mask, v)], dp[(prev_mask, u)] + shortest_paths[u][v])
    
    # Find the minimum weight among all possible ending vertices
    min_weight = math.inf
    full_mask = (1 << N) - 1
    for v in range(1, N+1):
        if (full_mask, v) in dp:  # Ensure the state exists
            min_weight = min(min_weight, dp[(full_mask, v)])
    
    return min_weight

if __name__ == "__main__":
    main()