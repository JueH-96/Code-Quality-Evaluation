from collections import defaultdict
import heapq
from typing import List

class Solution:
    def minimumCost(self, source: str, target: str, original: List[str], changed: List[str], cost: List[int]) -> int:
        # Build the graph
        graph = defaultdict(dict)
        for src, dst, cst in zip(original, changed, cost):
            if dst not in graph[src] or cst < graph[src][dst]:
                graph[src][dst] = cst
        
        # Function to find the shortest path from start to end using Dijkstra's algorithm
        def dijkstra(start, end):
            if start == end:
                return 0
            pq = [(0, start)]
            visited = set()
            while pq:
                current_cost, current_node = heapq.heappop(pq)
                if current_node == end:
                    return current_cost
                if current_node in visited:
                    continue
                visited.add(current_node)
                for neighbor, weight in graph[current_node].items():
                    if neighbor not in visited:
                        heapq.heappush(pq, (current_cost + weight, neighbor))
            return float('inf')
        
        total_cost = 0
        for src, dst in zip(source, target):
            if src != dst:
                cost_to_change = dijkstra(src, dst)
                if cost_to_change == float('inf'):
                    return -1
                total_cost += cost_to_change
        
        return total_cost