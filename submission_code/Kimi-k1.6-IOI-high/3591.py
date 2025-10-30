import heapq
from typing import List

class Solution:
    def shiftDistance(self, s: str, t: str, nextCost: List[int], previousCost: List[int]) -> int:
        # Precompute the cost matrix using Dijkstra's algorithm for each starting node
        cost_matrix = self.compute_cost_matrix(nextCost, previousCost)
        
        total_cost = 0
        for sc, tc in zip(s, t):
            u = ord(sc) - ord('a')
            v = ord(tc) - ord('a')
            total_cost += cost_matrix[u][v]
        
        return total_cost
    
    def compute_cost_matrix(self, nextCost: List[int], previousCost: List[int]) -> List[List[int]]:
        cost = [[float('inf')] * 26 for _ in range(26)]
        
        for u in range(26):
            # Dijkstra's algorithm for node u
            dist = [float('inf')] * 26
            dist[u] = 0
            heap = []
            heapq.heappush(heap, (0, u))
            
            while heap:
                current_dist, v = heapq.heappop(heap)
                if current_dist > dist[v]:
                    continue
                
                # Explore next node (shift to next character)
                next_node = (v + 1) % 26
                next_total = current_dist + nextCost[v]
                if dist[next_node] > next_total:
                    dist[next_node] = next_total
                    heapq.heappush(heap, (next_total, next_node))
                
                # Explore previous node (shift to previous character)
                prev_node = (v - 1) % 26
                prev_total = current_dist + previousCost[v]
                if dist[prev_node] > prev_total:
                    dist[prev_node] = prev_total
                    heapq.heappush(heap, (prev_total, prev_node))
            
            # Update the cost matrix for starting node u
            for v in range(26):
                cost[u][v] = dist[v]
        
        return cost