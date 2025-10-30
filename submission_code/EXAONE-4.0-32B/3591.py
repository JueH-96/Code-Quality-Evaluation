import heapq
from typing import List

class Solution:
    def shiftDistance(self, s: str, t: str, nextCost: List[int], previousCost: List[int]) -> int:
        n = len(s)
        INF = 10**18
        dist_matrix = [[INF] * 26 for _ in range(26)]
        
        for start in range(26):
            dist = [INF] * 26
            dist[start] = 0
            heap = [(0, start)]
            
            while heap:
                d, node = heapq.heappop(heap)
                if d != dist[node]:
                    continue
                neighbors = [
                    ((node + 1) % 26, nextCost[node]),
                    ((node - 1) % 26, previousCost[node])
                ]
                for neighbor, cost_val in neighbors:
                    new_dist = d + cost_val
                    if new_dist < dist[neighbor]:
                        dist[neighbor] = new_dist
                        heapq.heappush(heap, (new_dist, neighbor))
            dist_matrix[start] = dist
        
        total_cost = 0
        for i in range(n):
            u = ord(s[i]) - ord('a')
            v = ord(t[i]) - ord('a')
            total_cost += dist_matrix[u][v]
        
        return total_cost