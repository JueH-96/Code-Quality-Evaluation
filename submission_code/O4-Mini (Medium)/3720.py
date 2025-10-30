import heapq
from typing import List

class Solution:
    def minMaxWeight(self, n: int, edges: List[List[int]], threshold: int) -> int:
        # Build reversed graph: for each edge a->b with weight w,
        # add edge b->a of weight w in the reversed graph.
        adj = [[] for _ in range(n)]
        for a, b, w in edges:
            adj[b].append((a, w))
        
        # dist[v] = the minimum possible "bottleneck" (maximum-edge-on-path) 
        # from 0 to v in the reversed graph, which equals the
        # minimum possible bottleneck from v to 0 in the original graph.
        INF = 10**18
        dist = [INF] * n
        dist[0] = 0
        
        # Min-heap on the bottleneck cost
        heap = [(0, 0)]  # (current_bottleneck, node)
        
        while heap:
            cur_bottle, u = heapq.heappop(heap)
            if cur_bottle > dist[u]:
                continue
            # Relax edges in the reversed graph
            for v, w in adj[u]:
                nb = cur_bottle if cur_bottle > w else w
                if nb < dist[v]:
                    dist[v] = nb
                    heapq.heappush(heap, (nb, v))
        
        # If any node is unreachable, return -1
        if any(d == INF for d in dist):
            return -1
        
        # The answer is the maximum bottleneck among all nodes
        return max(dist)