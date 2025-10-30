from collections import defaultdict
from typing import List

class Solution:
    def minimumCost(self, source: str, target: str, original: List[str], changed: List[str], cost: List[int]) -> int:
        # Create a graph to store the minimum cost of transforming one character to another
        graph = defaultdict(lambda: defaultdict(lambda: float('inf')))
        for c, t, w in zip(original, changed, cost):
            graph[c][t] = min(graph[c][t], w)
        
        # Floyd-Warshall algorithm to find the shortest path between all pairs of characters
        for k in graph:
            for i in graph:
                for j in graph:
                    if graph[i][k] != float('inf') and graph[k][j] != float('inf'):
                        graph[i][j] = min(graph[i][j], graph[i][k] + graph[k][j])
        
        # Calculate the total cost to transform source to target
        total_cost = 0
        for s, t in zip(source, target):
            if s != t:
                if graph[s][t] == float('inf'):
                    return -1
                total_cost += graph[s][t]
        
        return total_cost