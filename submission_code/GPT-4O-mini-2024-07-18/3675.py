from typing import List
from collections import defaultdict

class Solution:
    def maximizeSumOfWeights(self, edges: List[List[int]], k: int) -> int:
        # Create an adjacency list to represent the tree
        graph = defaultdict(list)
        for u, v, w in edges:
            graph[u].append((v, w))
            graph[v].append((u, w))
        
        # To store the maximum weights we can keep
        total_weight = 0
        
        # To keep track of the number of edges connected to each node
        edge_count = [0] * len(graph)
        
        # Sort edges by weight in descending order
        edges.sort(key=lambda x: -x[2])
        
        # Iterate over edges and decide whether to keep them
        for u, v, w in edges:
            if edge_count[u] < k and edge_count[v] < k:
                total_weight += w
                edge_count[u] += 1
                edge_count[v] += 1
        
        return total_weight