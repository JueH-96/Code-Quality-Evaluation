from typing import List
from collections import defaultdict
import heapq

class Solution:
    def minimumCost(self, source: str, target: str, original: List[str], changed: List[str], cost: List[int]) -> int:
        # Create a graph where each character is a node
        graph = defaultdict(list)
        
        # Fill the graph with edges representing possible transformations and their costs
        for o, c, z in zip(original, changed, cost):
            graph[o].append((c, z))
        
        # Function to find the minimum cost to transform a character to another using Dijkstra's algorithm
        def dijkstra(start, end):
            if start == end:
                return 0
            # Priority queue for Dijkstra's algorithm
            pq = [(0, start)]
            # Dictionary to store the minimum cost to reach each character
            min_cost = {start: 0}
            
            while pq:
                current_cost, current_char = heapq.heappop(pq)
                
                if current_char == end:
                    return current_cost
                
                for neighbor, transformation_cost in graph[current_char]:
                    new_cost = current_cost + transformation_cost
                    if neighbor not in min_cost or new_cost < min_cost[neighbor]:
                        min_cost[neighbor] = new_cost
                        heapq.heappush(pq, (new_cost, neighbor))
            
            return float('inf')
        
        total_cost = 0
        
        # Calculate the total minimum cost to transform source to target
        for s_char, t_char in zip(source, target):
            if s_char != t_char:
                cost_to_transform = dijkstra(s_char, t_char)
                if cost_to_transform == float('inf'):
                    return -1
                total_cost += cost_to_transform
        
        return total_cost