from typing import List

class Solution:
    def shiftDistance(self, s: str, t: str, nextCost: List[int], previousCost: List[int]) -> int:
        # Initialize the cost matrix with infinity
        INF = float('inf')
        cost = [[INF] * 26 for _ in range(26)]
        
        # Cost from a character to itself is 0
        for i in range(26):
            cost[i][i] = 0
        
        # Set the direct shifting costs
        for c in range(26):
            next_char = (c + 1) % 26
            prev_char = (c - 1 + 26) % 26
            if nextCost[c] < cost[c][next_char]:
                cost[c][next_char] = nextCost[c]
            if previousCost[c] < cost[c][prev_char]:
                cost[c][prev_char] = previousCost[c]
        
        # Floyd-Warshall to compute all-pairs minimum cost
        for k in range(26):
            for i in range(26):
                for j in range(26):
                    if cost[i][k] + cost[k][j] < cost[i][j]:
                        cost[i][j] = cost[i][k] + cost[k][j]
        
        # Calculate the total minimum cost to transform s into t
        total_cost = 0
        for sc, tc in zip(s, t):
            a = ord(sc) - ord('a')
            b = ord(tc) - ord('a')
            total_cost += cost[a][b]
        
        return total_cost