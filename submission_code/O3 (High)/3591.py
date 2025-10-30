from typing import List

class Solution:
    def shiftDistance(self, s: str, t: str, nextCost: List[int], previousCost: List[int]) -> int:
        # number of letters
        ALPHA = 26
        INF = 10 ** 18           # large enough for 26 * 1e9
        
        # ---------------------------------------------------------------------
        # build the directed graph: each letter is a node,
        # edges go to the next / previous letter with their corresponding costs
        # ---------------------------------------------------------------------
        dist = [[INF] * ALPHA for _ in range(ALPHA)]
        for i in range(ALPHA):
            dist[i][i] = 0                                    # zero to itself
            dist[i][(i + 1) % ALPHA] = min(dist[i][(i + 1) % ALPHA], nextCost[i])
            dist[i][(i - 1) % ALPHA] = min(dist[i][(i - 1) % ALPHA], previousCost[i])
        
        # ---------------------------------------------------------------------
        # Floyd-Warshall – all-pairs shortest paths on 26 nodes / 52 edges
        # ---------------------------------------------------------------------
        for k in range(ALPHA):
            dk = dist[k]
            for i in range(ALPHA):
                dik = dist[i][k]
                if dik == INF:
                    continue
                di = dist[i]
                for j in range(ALPHA):
                    nd = dik + dk[j]
                    if nd < di[j]:
                        di[j] = nd
                        
        # ---------------------------------------------------------------------
        # accumulate the minimal cost for every position in the strings
        # ---------------------------------------------------------------------
        total = 0
        base = ord('a')
        for sc, tc in zip(s, t):
            total += dist[ord(sc) - base][ord(tc) - base]
        return total