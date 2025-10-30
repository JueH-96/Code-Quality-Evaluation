import sys
from collections import deque

# Use sys.stdin.readline for faster input
# input = sys.stdin.readline # No need for alias, use directly

def find_farthest(start_node, n, adj):
    """
    Performs BFS from start_node to find the node farthest away
    and its distance. Works on a tree with non-negative weights.
    """
    # Initialize distances. Use -1 to indicate unvisited.
    # Distances can be large, Python's int handles this.
    dist = [-1] * n
    dist[start_node] = 0
    
    queue = deque([start_node])
    
    farthest_node = start_node
    max_dist = 0 # Initialize max distance found so far
    
    while queue:
        u = queue.popleft()
        
        # Update farthest node found during this traversal
        # If current node u is farther than max_dist found so far
        if dist[u] > max_dist:
             max_dist = dist[u]
             farthest_node = u
             
        # Explore neighbors
        for v, weight in adj[u]:
            # If neighbor v is unvisited
            if dist[v] == -1:
                # Update distance to v
                dist[v] = dist[u] + weight
                # Add v to the queue
                queue.append(v)
    
    # Return the farthest node found and its distance from start_node
    return (farthest_node, max_dist)

# Read input
n = int(sys.stdin.readline())

# Adjacency list representation of the tree
# adj[i] is a list of tuples (neighbor_node, edge_weight)
# Use 0-indexed nodes internally (from 0 to n-1)
adj = [[] for _ in range(n)]

# Calculate the sum of all edge lengths
total_edge_length = 0 

# Read the edges and build the adjacency list
for _ in range(n - 1):
    # Read cities a, b (1-indexed) and cost c
    a, b, c = map(int, sys.stdin.readline().split())
    
    # Convert city numbers to 0-indexed node numbers
    u, v = a - 1, b - 1
    
    # Add edges to the adjacency list (tree is undirected)
    adj[u].append((v, c))
    adj[v].append((u, c))
    
    # Add edge cost to the total length
    total_edge_length += c

# The minimum traversal distance to visit all nodes in a tree
# is 2 * (sum of all edge weights) - diameter of the tree.

# Step 1: Find one endpoint of a diameter.
# Start a BFS from an arbitrary node (e.g., node 0, representing city 1)
# to find the node farthest away from it. This farthest node must be an endpoint of a diameter.
(one_endpoint_of_diameter, _) = find_farthest(0, n, adj)

# Step 2: Find the other endpoint of the diameter and its length.
# Start a BFS from the node found in Step 1. The node farthest away from this endpoint
# is the other endpoint of the diameter, and the distance is the diameter length.
(_, diameter) = find_farthest(one_endpoint_of_diameter, n, adj)

# Step 3: Calculate the minimum travel distance
# The formula is 2 * total_edge_length - diameter
min_travel_distance = 2 * total_edge_length - diameter

# Print the result
print(min_travel_distance)