from typing import List

class Solution:
    def minimumCost(self, source: str, target: str, original: List[str], changed: List[str], cost: List[int]) -> int:
        INF = 10**18
        dist = [[INF] * 26 for _ in range(26)]
        
        # Initialize distance to self as 0
        for i in range(26):
            dist[i][i] = 0
        
        # Set initial distances based on the available operations
        for o, c, co in zip(original, changed, cost):
            x = ord(o) - ord('a')
            y = ord(c) - ord('a')
            if dist[x][y] > co:
                dist[x][y] = co
        
        # Floyd-Warshall algorithm to compute all-pairs shortest paths
        for k in range(26):
            for i in range(26):
                if dist[i][k] == INF:
                    continue
                for j in range(26):
                    if dist[k][j] == INF:
                        continue
                    if dist[i][j] > dist[i][k] + dist[k][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
        
        total_cost = 0
        for s, t in zip(source, target):
            if s == t:
                continue
            x = ord(s) - ord('a')
            y = ord(t) - ord('a')
            if dist[x][y] == INF:
                return -1
            total_cost += dist[x][y]
        
        return total_cost