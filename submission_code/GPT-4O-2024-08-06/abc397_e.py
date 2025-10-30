# YOUR CODE HERE
def can_decompose_tree(N, K, edges):
    from collections import defaultdict
    
    # Calculate number of vertices
    num_vertices = N * K
    
    # Create adjacency list and degree count
    degree = [0] * (num_vertices + 1)
    
    # Fill the degree array
    for u, v in edges:
        degree[u] += 1
        degree[v] += 1
    
    # Count the number of leaf nodes (degree 1)
    leaf_count = sum(1 for d in degree if d == 1)
    
    # Check if the number of leaf nodes is exactly 2N
    if leaf_count != 2 * N:
        return "No"
    
    # Check if any vertex has degree greater than 2
    # If it does, it must be a junction point
    for d in degree:
        if d > 2:
            return "No"
    
    return "Yes"

import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
K = int(data[1])
edges = []

index = 2
for _ in range(N * K - 1):
    u = int(data[index])
    v = int(data[index + 1])
    edges.append((u, v))
    index += 2

print(can_decompose_tree(N, K, edges))