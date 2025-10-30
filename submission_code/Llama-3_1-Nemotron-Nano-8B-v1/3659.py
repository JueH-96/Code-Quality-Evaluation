from typing import List
from collections import defaultdict

MOD = 10**9 + 7

class Solution:
    def countPathsWithXorValue(self, grid: List[List[int]], k: int) -> int:
        m = len(grid)
        n = len(grid[0])
        dp = [[defaultdict(int) for _ in range(n)] for _ in range(m)]
        start_xor = grid[0][0]
        dp[0][0][start_xor] = 1
        
        for i in range(m):
            for j in range(n):
                if i == 0 and j == 0:
                    continue
                current = grid[i][j]
                current_dp = defaultdict(int)
                if i > 0:
                    for key, val in dp[i-1][j].items():
                        new_key = key ^ current
                        current_dp[new_key] = (current_dp[new_key] + val) % MOD
                if j > 0:
                    for key, val in dp[i][j-1].items():
                        new_key = key ^ current
                        current_dp[new_key] = (current_dp[new_key] + val) % MOD
                dp[i][j] = current_dp
        
        return dp[m-1][n-1].get(k, 0) % MOD