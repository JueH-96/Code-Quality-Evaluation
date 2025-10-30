from typing import List

class Solution:
    def minimumCost(self, source: str, target: str, original: List[str], changed: List[str], cost: List[int]) -> int:
        INF = float('inf')
        # Initialize distance matrix for 26 lowercase letters
        dist = [[INF] * 26 for _ in range(26)]
        for i in range(26):
            dist[i][i] = 0  # Cost to self is zero
        
        # Populate the distance matrix with direct edges
        for o_char, c_char, co in zip(original, changed, cost):
            u = ord(o_char) - ord('a')
            v = ord(c_char) - ord('a')
            if dist[u][v] > co:
                dist[u][v] = co
        
        # Floyd-Warshall algorithm to find all pairs shortest paths
        for k in range(26):
            for i in range(26):
                for j in range(26):
                    if dist[i][k] + dist[k][j] < dist[i][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
        
        total_cost = 0
        for s_char, t_char in zip(source, target):
            if s_char == t_char:
                continue  # No cost needed
            u = ord(s_char) - ord('a')
            v = ord(t_char) - ord('a')
            if dist[u][v] == INF:
                return -1  # No possible transformation
            total_cost += dist[u][v]
        
        return total_cost