from typing import List
import heapq

class Solution:
    def minMaxWeight(self, n: int, edges: List[List[int]], threshold: int) -> int:
        # Build reversed adjacency list: for each original edge u->v with weight w,
        # add edge v->u in the reversed graph.
        rev_adj = [[] for _ in range(n)]
        for u, v, w in edges:
            rev_adj[v].append((u, w))
        
        # dist[i] will hold the minimum possible "maximum edge weight" along
        # any path from node i to node 0 in the original graph.
        # Equivalently, in the reversed graph, it's the minimax cost from 0 to i.
        INF = 10**18
        dist = [INF] * n
        dist[0] = 0
        
        # Min-heap for Dijkstra-style minimax path search.
        # Each entry is (current_minimax_cost, node).
        heap = [(0, 0)]
        
        while heap:
            curr_cost, u = heapq.heappop(heap)
            if curr_cost > dist[u]:
                continue
            # Relax edges in the reversed graph
            for v, w in rev_adj[u]:
                # If we go from 0->...->u->v (in reversed), the path's
                # bottleneck is max(curr_cost, w).
                new_cost = max(curr_cost, w)
                if new_cost < dist[v]:
                    dist[v] = new_cost
                    heapq.heappush(heap, (new_cost, v))
        
        # The answer is the maximum dist[v] over all nodes v.
        # If any node is unreachable, return -1.
        answer = 0
        for d in dist:
            if d == INF:
                return -1
            if d > answer:
                answer = d
        
        return answer