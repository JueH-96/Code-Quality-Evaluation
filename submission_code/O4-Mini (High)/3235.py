from typing import List

class Solution:
    def minimumCost(self, source: str, target: str, original: List[str], changed: List[str], cost: List[int]) -> int:
        # Number of nodes in our graph (one per lowercase letter)
        N = 26
        INF = 10**18
        
        # Initialize distance matrix
        dist = [[INF] * N for _ in range(N)]
        for i in range(N):
            dist[i][i] = 0
        
        # Add the given directed edges with their costs
        for x, y, c in zip(original, changed, cost):
            u = ord(x) - ord('a')
            v = ord(y) - ord('a')
            if c < dist[u][v]:
                dist[u][v] = c
        
        # Floyd–Warshall to compute all‑pairs shortest paths
        for k in range(N):
            for i in range(N):
                if dist[i][k] == INF:
                    continue
                for j in range(N):
                    nd = dist[i][k] + dist[k][j]
                    if nd < dist[i][j]:
                        dist[i][j] = nd
        
        # Sum up the cost for each character conversion in source -> target
        total = 0
        for sc, tc in zip(source, target):
            if sc == tc:
                continue
            u = ord(sc) - ord('a')
            v = ord(tc) - ord('a')
            d = dist[u][v]
            if d == INF:
                return -1
            total += d
        
        return total