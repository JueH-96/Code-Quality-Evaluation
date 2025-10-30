import collections
from typing import List

class Solution:
    def timeTaken(self, edges: List[List[int]]) -> List[int]:
        n = len(edges) + 1
        # Weight: 2 if even, 1 if odd
        weight = [2 if i % 2 == 0 else 1 for i in range(n)]
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Helper function to compute cumulative weight distances from a start node
        def compute_dist(start):
            dist = [-1] * n
            dist[start] = weight[start]
            queue = collections.deque([start])
            while queue:
                node = queue.popleft()
                for nei in adj[node]:
                    if dist[nei] == -1:  # not visited
                        dist[nei] = dist[node] + weight[nei]
                        queue.append(nei)
            return dist
        
        # Find one end of the diameter
        dist_from_0 = compute_dist(0)
        A = max(range(n), key=lambda x: dist_from_0[x])
        
        # Find the other end of the diameter
        dist_from_A = compute_dist(A)
        B = max(range(n), key=lambda x: dist_from_A[x])
        
        # Compute distances from A and B
        dist_from_A = compute_dist(A)  # cum_w(A, u) for all u
        dist_from_B = compute_dist(B)  # cum_w(B, u) for all u
        
        # Compute the time for each starting node
        times = []
        for s in range(n):
            max_cum_w = max(dist_from_A[s], dist_from_B[s])
            time_s = max_cum_w - weight[s]
            times.append(time_s)
        
        return times