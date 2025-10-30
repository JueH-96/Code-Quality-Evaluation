from itertools import combinations
from collections import deque

def max_shortest_path(N, M, K, edges):
    # Create adjacency list
    adj_list = [[] for _ in range(N + 1)]
    for idx, (u, v) in enumerate(edges):
        adj_list[u].append((v, idx))
    
    # Function to calculate shortest path from 1 to N with given edge weights
    def shortest_path(edge_weights):
        dist = [float('inf')] * (N + 1)
        dist[1] = 0
        queue = deque([1])
        
        while queue:
            node = queue.popleft()
            current_dist = dist[node]
            
            for neighbor, edge_idx in adj_list[node]:
                weight = edge_weights[edge_idx]
                if current_dist + weight < dist[neighbor]:
                    dist[neighbor] = current_dist + weight
                    queue.append(neighbor)
        
        return dist[N]
    
    # Initial edge weights (all 0)
    initial_weights = [0] * M
    
    # Try all combinations of K edges to set to weight 1
    max_distance = 0
    for comb in combinations(range(M), K):
        # Create a copy of initial weights
        edge_weights = initial_weights[:]
        # Set selected edges to weight 1
        for idx in comb:
            edge_weights[idx] = 1
        
        # Calculate the shortest path with these weights
        distance = shortest_path(edge_weights)
        # Update the maximum distance found
        max_distance = max(max_distance, distance)
    
    return max_distance

# Reading input
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
K = int(data[2])

edges = []
index = 3
for _ in range(M):
    u = int(data[index])
    v = int(data[index + 1])
    edges.append((u, v))
    index += 2

# Solve the problem
result = max_shortest_path(N, M, K, edges)
print(result)