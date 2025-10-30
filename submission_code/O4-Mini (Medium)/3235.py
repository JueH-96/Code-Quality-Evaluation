from typing import List

class Solution:
    def minimumCost(self, source: str, target: str, original: List[str], changed: List[str], cost: List[int]) -> int:
        # Number of letters in the alphabet
        ALPH = 26
        INF = 10**18
        
        # Build distance matrix for transformations between letters
        # dist[i][j] = minimum cost to go from letter i to letter j
        dist = [[INF] * ALPH for _ in range(ALPH)]
        
        # Zero cost to stay the same letter
        for i in range(ALPH):
            dist[i][i] = 0
        
        # Initialize edges from the given operations
        for x, y, c in zip(original, changed, cost):
            u = ord(x) - ord('a')
            v = ord(y) - ord('a')
            # If multiple operations exist, keep the cheapest
            if c < dist[u][v]:
                dist[u][v] = c
        
        # Floyd–Warshall to compute all-pairs shortest paths
        for k in range(ALPH):
            for i in range(ALPH):
                # Skip if i->k is unreachable
                if dist[i][k] == INF:
                    continue
                dik = dist[i][k]
                row_i = dist[i]
                row_k = dist[k]
                for j in range(ALPH):
                    # Relax via k
                    alt = dik + row_k[j]
                    if alt < row_i[j]:
                        row_i[j] = alt
        
        # Now compute the total cost to transform source to target
        total_cost = 0
        for sc, tc in zip(source, target):
            if sc == tc:
                continue
            u = ord(sc) - ord('a')
            v = ord(tc) - ord('a')
            c = dist[u][v]
            if c == INF:
                return -1
            total_cost += c
        
        return total_cost