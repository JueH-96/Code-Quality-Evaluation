from typing import List
import heapq

class Solution:
    def shiftDistance(self, s: str, t: str, nextCost: List[int], previousCost: List[int]) -> int:
        # Build a directed weighted graph on the 26 letters (0 = 'a', 25 = 'z').
        # From each letter u you can go to (u+1)%26 at cost nextCost[u],
        # or to (u-1)%26 at cost previousCost[u].
        adj = [[] for _ in range(26)]
        for u in range(26):
            adj[u].append(((u + 1) % 26, nextCost[u]))
            adj[u].append(((u - 1) % 26, previousCost[u]))
        
        # Precompute shortest path distances dist[u][v] for all u,v in [0..25]
        # using Dijkstra from each u.
        INF = 10**30
        dist = [[INF]*26 for _ in range(26)]
        for start in range(26):
            d = dist[start]
            d[start] = 0
            heap = [(0, start)]
            while heap:
                cd, u = heapq.heappop(heap)
                if cd > d[u]:
                    continue
                for v, w in adj[u]:
                    nd = cd + w
                    if nd < d[v]:
                        d[v] = nd
                        heapq.heappush(heap, (nd, v))
        
        # Sum up the cost to transform each s[i] -> t[i]
        total = 0
        for cs, ct in zip(s, t):
            u = ord(cs) - ord('a')
            v = ord(ct) - ord('a')
            total += dist[u][v]
        
        return total