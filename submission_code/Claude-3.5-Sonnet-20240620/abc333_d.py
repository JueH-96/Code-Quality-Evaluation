# YOUR CODE HERE
from collections import defaultdict

def min_operations_to_delete_vertex_1(N, edges):
    # Create an adjacency list representation of the tree
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)
    
    # Function to perform DFS and count leaves
    def dfs(node, parent):
        if len(graph[node]) == 1 and graph[node][0] == parent:
            return 1  # This is a leaf
        
        leaf_count = 0
        for child in graph[node]:
            if child != parent:
                leaf_count += dfs(child, node)
        
        return leaf_count
    
    # If vertex 1 is a leaf, we can delete it in one operation
    if len(graph[1]) == 1:
        return 1
    
    # Count the number of leaves in each subtree of vertex 1
    total_leaves = 0
    for child in graph[1]:
        total_leaves += dfs(child, 1)
    
    # The answer is the total number of leaves
    return total_leaves

# Read input
N = int(input())
edges = [list(map(int, input().split())) for _ in range(N-1)]

# Compute and print the result
result = min_operations_to_delete_vertex_1(N, edges)
print(result)