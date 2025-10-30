from typing import List

class Solution:
    def shiftDistance(self, s: str, t: str, nextCost: List[int], previousCost: List[int]) -> int:
        # Initialize the distance matrix with infinity
        INF = float('inf')
        dist = [[INF] * 26 for _ in range(26)]
        for i in range(26):
            dist[i][i] = 0
            # Set next shift edge
            next_node = (i + 1) % 26
            dist[i][next_node] = min(dist[i][next_node], nextCost[i])
            # Set previous shift edge
            prev_node = (i - 1) % 26
            dist[i][prev_node] = min(dist[i][prev_node], previousCost[i])
        
        # Floyd-Warshall algorithm to compute all pairs shortest paths
        for k in range(26):
            for i in range(26):
                for j in range(26):
                    if dist[i][j] > dist[i][k] + dist[k][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
        
        # Calculate the total cost
        total = 0
        for sc, tc in zip(s, t):
            i = ord(sc) - ord('a')
            j = ord(tc) - ord('a')
            total += dist[i][j]
        
        return total