import heapq
from typing import List

class Solution:
    def shiftDistance(self, s: str, t: str, nextCost: List[int], previousCost: List[int]) -> int:
        # Precompute the cost matrix using Dijkstra's algorithm for each starting character
        cost = [[0] * 26 for _ in range(26)]
        for start in range(26):
            # Initialize distances for Dijkstra's algorithm
            dist = [float('inf')] * 26
            dist[start] = 0
            heap = []
            heapq.heappush(heap, (0, start))
            
            while heap:
                current_cost, u = heapq.heappop(heap)
                if current_cost > dist[u]:
                    continue
                # Explore next character
                v_next = (u + 1) % 26
                new_cost = current_cost + nextCost[u]
                if new_cost < dist[v_next]:
                    dist[v_next] = new_cost
                    heapq.heappush(heap, (new_cost, v_next))
                # Explore previous character
                v_prev = (u - 1) % 26
                new_cost = current_cost + previousCost[u]
                if new_cost < dist[v_prev]:
                    dist[v_prev] = new_cost
                    heapq.heappush(heap, (new_cost, v_prev))
            # Update the cost matrix for the starting character
            for end in range(26):
                cost[start][end] = dist[end]
        
        # Calculate the total shift distance by summing up the costs for each character pair
        total_cost = 0
        for sc, tc in zip(s, t):
            c1 = ord(sc) - ord('a')
            c2 = ord(tc) - ord('a')
            total_cost += cost[c1][c2]
        
        return total_cost