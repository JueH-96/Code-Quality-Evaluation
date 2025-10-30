from typing import List
import heapq
from collections import defaultdict, deque

class Solution:
    def minimumCost(self, source: str, target: str, original: List[str], changed: List[str], cost: List[int]) -> int:
        # Create a graph for the transformations
        graph = defaultdict(list)
        for o, c, z in zip(original, changed, cost):
            graph[o].append((c, z))
        
        # Function to find the minimum cost to transform a character
        def dijkstra(start, end):
            if start == end:
                return 0
            pq = [(0, start)]
            visited = set()
            while pq:
                current_cost, current_char = heapq.heappop(pq)
                if current_char in visited:
                    continue
                visited.add(current_char)
                if current_char == end:
                    return current_cost
                for neighbor, weight in graph[current_char]:
                    if neighbor not in visited:
                        heapq.heappush(pq, (current_cost + weight, neighbor))
            return float('inf')
        
        total_cost = 0
        for s_char, t_char in zip(source, target):
            if s_char != t_char:
                min_cost = dijkstra(s_char, t_char)
                if min_cost == float('inf'):
                    return -1
                total_cost += min_cost
        
        return total_cost