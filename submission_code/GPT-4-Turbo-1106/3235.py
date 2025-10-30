from typing import List
from collections import defaultdict
import heapq

class Solution:
    def minimumCost(self, source: str, target: str, original: List[str], changed: List[str], cost: List[int]) -> int:
        # Create a dictionary to store the minimum cost to change each character to another
        change_cost = defaultdict(lambda: defaultdict(lambda: float('inf')))
        
        # Populate the dictionary with the given costs
        for o, c, co in zip(original, changed, cost):
            change_cost[o][c] = min(change_cost[o][c], co)
        
        # Use Dijkstra's algorithm to find the minimum cost to change each character to another
        for o in change_cost:
            min_heap = [(0, o)]
            while min_heap:
                current_cost, current_char = heapq.heappop(min_heap)
                for next_char in change_cost[current_char]:
                    next_cost = current_cost + change_cost[current_char][next_char]
                    if next_cost < change_cost[o][next_char]:
                        change_cost[o][next_char] = next_cost
                        heapq.heappush(min_heap, (next_cost, next_char))
        
        # Calculate the total cost to convert source to target
        total_cost = 0
        for s, t in zip(source, target):
            if s == t:
                continue
            if change_cost[s][t] == float('inf'):
                return -1
            total_cost += change_cost[s][t]
        
        return total_cost