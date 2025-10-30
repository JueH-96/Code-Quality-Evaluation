from typing import List

class Solution:
    def minimumCost(self, source: str, target: str, original: List[str], changed: List[str], cost: List[int]) -> int:
        # Number of lowercase English letters
        ALPHABET_SIZE = 26
        INF = 10**30
        
        # Initialize distance matrix for Floyd-Warshall
        # dist[u][v] = minimum cost to change char u -> char v
        dist = [[INF] * ALPHABET_SIZE for _ in range(ALPHABET_SIZE)]
        for i in range(ALPHABET_SIZE):
            dist[i][i] = 0
        
        # Populate direct-change edges, taking the minimum if multiple edges exist
        m = len(original)
        for j in range(m):
            u = ord(original[j]) - ord('a')
            v = ord(changed[j]) - ord('a')
            c = cost[j]
            if c < dist[u][v]:
                dist[u][v] = c
        
        # Floyd-Warshall to compute all-pairs shortest paths
        for k in range(ALPHABET_SIZE):
            for i in range(ALPHABET_SIZE):
                # skip if unreachable
                if dist[i][k] == INF:
                    continue
                row_ik = dist[i][k]
                for j in range(ALPHABET_SIZE):
                    nd = row_ik + dist[k][j]
                    if nd < dist[i][j]:
                        dist[i][j] = nd
        
        # Now accumulate the cost to convert source -> target
        total_cost = 0
        n = len(source)
        for i in range(n):
            s = ord(source[i]) - ord('a')
            t = ord(target[i]) - ord('a')
            # If already equal, no cost
            if s == t:
                continue
            # If no valid sequence of changes, impossible
            if dist[s][t] >= INF:
                return -1
            total_cost += dist[s][t]
        
        return total_cost