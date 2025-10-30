from typing import List
import math

class Solution:
    def minimumCost(self, source: str, target: str, original: List[str], changed: List[str], cost: List[int]) -> int:
        # There are 26 letters
        INF = math.inf
        n_letters = 26
        
        # initialize dp table: dp[u][v] = minimum cost to convert letter u to letter v
        dp = [[INF] * n_letters for _ in range(n_letters)]
        
        # For identity conversion, cost is 0
        for i in range(n_letters):
            dp[i][i] = 0
        
        # For each conversion rule update the direct cost
        for orig, new, c in zip(original, changed, cost):
            u = ord(orig) - ord('a')
            v = ord(new) - ord('a')
            dp[u][v] = min(dp[u][v], c)
        
        # Use Floyd Warshall to compute minimum cost conversion for each pair of letters.
        for k in range(n_letters):
            for i in range(n_letters):
                for j in range(n_letters):
                    if dp[i][k] + dp[k][j] < dp[i][j]:
                        dp[i][j] = dp[i][k] + dp[k][j]
                        
        # Compute total cost to convert source to target letter by letter.
        total_cost = 0
        n = len(source)
        for i in range(n):
            u = ord(source[i]) - ord('a')
            v = ord(target[i]) - ord('a')
            if dp[u][v] == INF:
                return -1
            total_cost += dp[u][v]
        return total_cost