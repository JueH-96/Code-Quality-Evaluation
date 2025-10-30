from typing import List

class Solution:
    def shiftDistance(self, s: str, t: str, nextCost: List[int], previousCost: List[int]) -> int:
        # There are 26 letters.
        n = 26
        INF = 10**18
        
        # Create a 26x26 matrix to store the minimum cost to go from one letter to another.
        dist = [[INF] * n for _ in range(n)]
        for i in range(n):
            dist[i][i] = 0
        
        # For each letter (represented as index i), add the direct transition costs:
        # - Moving to the next letter: cost = nextCost[i]
        # - Moving to the previous letter: cost = previousCost[i]
        for i in range(n):
            next_letter = (i + 1) % n
            prev_letter = (i - 1) % n
            dist[i][next_letter] = min(dist[i][next_letter], nextCost[i])
            dist[i][prev_letter] = min(dist[i][prev_letter], previousCost[i])
        
        # Use Floyd Warshall to compute the minimum cost between every pair of letters.
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dist[i][j] > dist[i][k] + dist[k][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
        
        # Now go through both strings letter by letter and sum the minimal cost to transform s[i] into t[i]
        total_cost = 0
        for sc, tc in zip(s, t):
            from_idx = ord(sc) - ord('a')
            to_idx = ord(tc) - ord('a')
            total_cost += dist[from_idx][to_idx]
        
        return total_cost