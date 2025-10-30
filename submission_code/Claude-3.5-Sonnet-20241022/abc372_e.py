from collections import defaultdict
import heapq

# Read input
N, Q = map(int, input().split())

# Initialize graph as adjacency list
graph = defaultdict(set)

# Process queries
for _ in range(Q):
    query = list(map(int, input().split()))
    
    if query[0] == 1:
        # Add edge
        _, u, v = query
        graph[u].add(v)
        graph[v].add(u)
        
    else:
        # Find kth largest neighbor
        _, v, k = query
        neighbors = list(graph[v])
        neighbors.append(v)  # Add vertex itself as it's connected to itself
        
        if len(neighbors) < k:
            print(-1)
        else:
            # Sort in descending order and get kth element (1-based index)
            neighbors.sort(reverse=True)
            print(neighbors[k-1])