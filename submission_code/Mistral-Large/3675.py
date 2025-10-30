from typing import List
from collections import defaultdict
import heapq

class Solution:
    def maximizeSumOfWeights(self, edges: List[List[int]], k: int) -> int:
        # Build the graph
        graph = defaultdict(list)
        for u, v, w in edges:
            graph[u].append((w, v))
            graph[v].append((w, u))

        # Sort edges by weight in descending order
        for node in graph:
            graph[node].sort(reverse=True)

        # Remove excess edges
        total_weight = 0
        for node in graph:
            if len(graph[node]) > k:
                # Keep only the top k edges
                graph[node] = graph[node][:k]

        # Calculate the sum of the remaining edges
        visited = set()
        for node in graph:
            for weight, neighbor in graph[node]:
                if (node, neighbor) not in visited and (neighbor, node) not in visited:
                    total_weight += weight
                    visited.add((node, neighbor))
                    visited.add((neighbor, node))

        return total_weight