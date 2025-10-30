from collections import defaultdict, deque
from heapq import heappop, heappush
from typing import List

class Solution:
    def minimumCost(self, source: str, target: str, original: List[str], changed: List[str], cost: List[int]) -> int:
        # Create a graph to represent the cost of changing characters
        graph = defaultdict(list)
        for o, c, co in zip(original, changed, cost):
            graph[o].append((c, co))

        # Use Dijkstra's algorithm to find the minimum cost to convert each character
        def dijkstra(start, end):
            if start == end:
                return 0
            min_heap = [(0, start)]
            visited = set()
            while min_heap:
                current_cost, current_char = heappop(min_heap)
                if current_char in visited:
                    continue
                visited.add(current_char)
                if current_char == end:
                    return current_cost
                for neighbor, edge_cost in graph[current_char]:
                    heappush(min_heap, (current_cost + edge_cost, neighbor))
            return float('inf')

        total_cost = 0
        for s, t in zip(source, target):
            conversion_cost = dijkstra(s, t)
            if conversion_cost == float('inf'):
                return -1
            total_cost += conversion_cost

        return total_cost