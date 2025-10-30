import heapq
from collections import defaultdict
from typing import List

class Solution:
    def minimumCost(self, source: str, target: str, original: List[str], changed: List[str], cost: List[int]) -> int:
        # Build the adjacency list with minimal direct edge costs
        adj = defaultdict(dict)
        for j in range(len(original)):
            x = original[j]
            y = changed[j]
            if x == y:
                continue  # No cost for same character
            current_cost = cost[j]
            if y not in adj[x] or adj[x][y] > current_cost:
                adj[x][y] = current_cost
        
        # Precompute the minimal distances between all pairs of characters using Dijkstra's algorithm
        INF = float('inf')
        dist = [[INF] * 26 for _ in range(26)]
        for x in range(26):
            char_x = chr(ord('a') + x)
            dist[x][x] = 0
            heap = []
            heapq.heappush(heap, (0, x))
            while heap:
                current_dist, u = heapq.heappop(heap)
                if current_dist > dist[x][u]:
                    continue
                # Explore all neighbors of the current character
                for v_char, cost_uv in adj.get(char_x, {}).items():
                    v = ord(v_char) - ord('a')
                    if dist[x][v] > current_dist + cost_uv:
                        dist[x][v] = current_dist + cost_uv
                        heapq.heappush(heap, (dist[x][v], v))
        
        # Calculate the total minimum cost
        total = 0
        for s_char, t_char in zip(source, target):
            s = ord(s_char) - ord('a')
            t = ord(t_char) - ord('a')
            if s != t:
                if dist[s][t] == INF:
                    return -1
                total += dist[s][t]
        return total