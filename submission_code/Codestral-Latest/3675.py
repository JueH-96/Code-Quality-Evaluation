from typing import List
import heapq

class Solution:
    def maximizeSumOfWeights(self, edges: List[List[int]], k: int) -> int:
        from collections import defaultdict, deque

        # Build the graph
        graph = defaultdict(list)
        for u, v, w in edges:
            graph[u].append((v, w))
            graph[v].append((u, w))

        # Calculate the degree of each node
        degree = [0] * len(graph)
        for u, v, w in edges:
            degree[u] += 1
            degree[v] += 1

        # Use a max-heap to store the edges sorted by weight
        max_heap = []
        for u, v, w in edges:
            heapq.heappush(max_heap, (-w, u, v))

        # Initialize the sum of weights
        total_weight = sum(w for u, v, w in edges)

        # Process the edges to ensure each node has at most k edges
        while max_heap:
            w, u, v = heapq.heappop(max_heap)
            w = -w
            if degree[u] > k or degree[v] > k:
                total_weight -= w
                degree[u] -= 1
                degree[v] -= 1
            else:
                break

        return total_weight