from typing import List
import heapq

class Solution:
    def maximizeSumOfWeights(self, edges: List[List[int]], k: int) -> int:
        # Create adjacency list
        adj = {}
        for u, v, w in edges:
            if u not in adj:
                adj[u] = []
            if v not in adj:
                adj[v] = []
            adj[u].append((w, v))
            adj[v].append((w, u))
        
        # Initialize the total sum of weights
        total_sum = sum(w for _, _, w in edges)
        
        # Priority queue to keep track of edges to remove
        pq = []
        
        # Traverse each node's adjacency list
        for node in adj:
            if len(adj[node]) > k:
                # Sort the edges by weight in ascending order
                adj[node].sort()
                # Push the smallest weights to the priority queue
                for i in range(len(adj[node]) - k):
                    heapq.heappush(pq, adj[node][i][0])
        
        # Remove the smallest weights until the priority queue is empty
        while pq:
            total_sum -= heapq.heappop(pq)
        
        return total_sum