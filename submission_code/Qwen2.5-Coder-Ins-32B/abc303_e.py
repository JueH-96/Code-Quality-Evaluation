# YOUR CODE HERE
import sys
from collections import defaultdict

def find_stars(N, edges):
    # Create adjacency list
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Find leaves (nodes with degree 1)
    leaves = [node for node in adj if len(adj[node]) == 1]
    
    # Initialize levels of stars
    levels = []
    
    # Process leaves to find stars
    while leaves:
        new_leaves = []
        current_level = 0
        visited = set()
        
        for leaf in leaves:
            if leaf in visited:
                continue
            visited.add(leaf)
            current_node = leaf
            current_level = 0
            
            # Traverse to the center of the star
            while len(adj[current_node]) == 2:
                visited.add(current_node)
                next_node = adj[current_node][0] if adj[current_node][0] not in visited else adj[current_node][1]
                adj[next_node].remove(current_node)
                current_node = next_node
                current_level += 1
            
            # Add the center node to visited
            visited.add(current_node)
            
            # If the center node has more than 2 connections, it's the center of a star
            if len(adj[current_node]) > 1:
                current_level += 1
                levels.append(current_level)
            
            # Collect new leaves
            for node in adj:
                if len(adj[node]) == 1 and node not in visited:
                    new_leaves.append(node)
        
        leaves = new_leaves
    
    # Sort levels and print
    levels.sort()
    print(" ".join(map(str, levels)))

# Read input
input = sys.stdin.read().split()
N = int(input[0])
edges = [(int(input[2*i+1]), int(input[2*i+2])) for i in range(N-1)]

# Find and print star levels
find_stars(N, edges)