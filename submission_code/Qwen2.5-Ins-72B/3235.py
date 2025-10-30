from collections import defaultdict
import heapq

class Solution:
    def minimumCost(self, source: str, target: str, original: List[str], changed: List[str], cost: List[int]) -> int:
        # Build the graph
        graph = defaultdict(list)
        for o, c, co in zip(original, changed, cost):
            graph[o].append((c, co))
        
        # Dijkstra's algorithm to find the minimum cost to convert one character to another
        def dijkstra(start, end):
            if start == end:
                return 0
            min_cost = {char: float('inf') for char in 'abcdefghijklmnopqrstuvwxyz'}
            min_cost[start] = 0
            heap = [(0, start)]
            while heap:
                current_cost, current_char = heapq.heappop(heap)
                if current_char == end:
                    return current_cost
                for next_char, next_cost in graph[current_char]:
                    if current_cost + next_cost < min_cost[next_char]:
                        min_cost[next_char] = current_cost + next_cost
                        heapq.heappush(heap, (current_cost + next_cost, next_char))
            return float('inf')
        
        # Calculate the total cost to convert source to target
        total_cost = 0
        for s, t in zip(source, target):
            if s != t:
                conversion_cost = dijkstra(s, t)
                if conversion_cost == float('inf'):
                    return -1
                total_cost += conversion_cost
        
        return total_cost