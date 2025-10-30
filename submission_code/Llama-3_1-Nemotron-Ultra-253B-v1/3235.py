from typing import List

class Solution:
    def minimumCost(self, source: str, target: str, original: List[str], changed: List[str], cost: List[int]) -> int:
        INF = float('inf')
        # Initialize distance matrix with infinity and zero for self transitions
        dist = [[INF] * 26 for _ in range(26)]
        for i in range(26):
            dist[i][i] = 0
        
        # Update direct transformation costs
        for j in range(len(original)):
            x = original[j]
            y = changed[j]
            z = cost[j]
            x_ord = ord(x) - ord('a')
            y_ord = ord(y) - ord('a')
            if dist[x_ord][y_ord] > z:
                dist[x_ord][y_ord] = z
        
        # Floyd-Warshall algorithm to find all pairs shortest paths
        for k in range(26):
            for i in range(26):
                for j in range(26):
                    if dist[i][k] + dist[k][j] < dist[i][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
        
        total_cost = 0
        for s_char, t_char in zip(source, target):
            if s_char == t_char:
                continue
            s_ord = ord(s_char) - ord('a')
            t_ord = ord(t_char) - ord('a')
            if dist[s_ord][t_ord] == INF:
                return -1
            total_cost += dist[s_ord][t_ord]
        
        return total_cost