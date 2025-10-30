from typing import List

class Solution:
    def minimumCost(self, source: str, target: str,
                    original: List[str], changed: List[str],
                    cost:    List[int]) -> int:
        
        # length mismatch cannot happen according to statement,
        # but guard just in case
        if len(source) != len(target):
            return -1
        
        ALPHA = 26                     # number of lowercase letters
        INF   = 10**18                 # big enough
        
        # distance matrix, dist[u][v] = minimum cost to go u -> v
        dist = [[INF]*ALPHA for _ in range(ALPHA)]
        for i in range(ALPHA):
            dist[i][i] = 0             # zero to stay in same letter
        
        # insert the given direct transformations (keep the cheapest)
        for o, c, w in zip(original, changed, cost):
            u = ord(o) - 97
            v = ord(c) - 97
            if w < dist[u][v]:
                dist[u][v] = w
        
        # Floyd-Warshall: all-pairs shortest paths on 26-node graph
        for k in range(ALPHA):
            for i in range(ALPHA):
                if dist[i][k] == INF:
                    continue
                for j in range(ALPHA):
                    if dist[k][j] == INF:
                        continue
                    new_d = dist[i][k] + dist[k][j]
                    if new_d < dist[i][j]:
                        dist[i][j] = new_d
        
        # compute total cost for every position
        total = 0
        for sc, tc in zip(source, target):
            if sc == tc:
                continue
            d = dist[ord(sc)-97][ord(tc)-97]
            if d == INF:               # impossible for this position
                return -1
            total += d
        
        return total