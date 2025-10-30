def find_max_alkane_subgraph(n, edges):
    from collections import defaultdict
    
    # Calculate degrees of each vertex
    degree = defaultdict(int)
    adjacency_list = defaultdict(list)
    
    for a, b in edges:
        degree[a] += 1
        degree[b] += 1
        adjacency_list[a].append(b)
        adjacency_list[b].append(a)
    
    # Find all vertices with degree 4
    potential_centers = [v for v in range(1, n + 1) if degree[v] == 4]
    
    if not potential_centers:
        return -1
    
    max_size = -1
    
    # For each potential center, try to form the largest alkane subgraph
    for center in potential_centers:
        # Check the neighbors of the center
        neighbors = adjacency_list[center]
        
        # Count how many of its neighbors are leaves (degree 1)
        leaf_count = sum(1 for neighbor in neighbors if degree[neighbor] == 1)
        
        # The center itself and its leaves form a subgraph
        # The size of this subgraph is the center + its leaves
        if leaf_count >= 4:
            # We can form an alkane with the center and 4 leaves
            max_size = max(max_size, 1 + leaf_count)
    
    return max_size if max_size != -1 else -1

# Reading input
import sys
input = sys.stdin.read
data = input().strip().split()
n = int(data[0])
edges = [(int(data[i]), int(data[i+1])) for i in range(1, len(data), 2)]

# Output the result
print(find_max_alkane_subgraph(n, edges))