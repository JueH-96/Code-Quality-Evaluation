import math
from typing import List

class Solution:
    def minimumCost(self, source: str, target: str, original: List[str], changed: List[str], cost: List[int]) -> int:
        INF = 2000000000  # A large number to represent infinity
        
        # Create a 26x26 distance matrix initialized to INF
        dist = [[INF for _ in range(26)] for _ in range(26)]
        
        # Set distance from a node to itself as 0
        for i in range(26):
            dist[i][i] = 0
        
        # Add all direct transformations with the minimum cost
        for orig_char, chan_char, c in zip(original, changed, cost):
            u = ord(orig_char) - ord('a')
            v = ord(chan_char) - ord('a')
            dist[u][v] = min(dist[u][v], c)
        
        # Run Floyd-Warshall algorithm to find shortest paths between all pairs
        for k in range(26):
            for i in range(26):
                for j in range(26):
                    if dist[i][k] < INF and dist[k][j] < INF:
                        dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
        
        # Calculate the total cost for transforming source to target
        total_cost = 0
        for s_char, t_char in zip(source, target):
            u = ord(s_char) - ord('a')
            v = ord(t_char) - ord('a')
            if dist[u][v] >= INF:
                return -1  # Impossible to transform this character
            total_cost += dist[u][v]
        
        return total_cost