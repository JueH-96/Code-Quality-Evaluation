from typing import List
from collections import defaultdict

class Solution:
    def countPathsWithXorValue(self, grid: List[List[int]], k: int) -> int:
        MOD = 10**9 + 7
        m = len(grid)
        n = len(grid[0]) if m > 0 else 0
        dp = [[defaultdict(int) for _ in range(n)] for _ in range(m)]
        
        # Initialize the starting cell
        dp[0][0][grid[0][0]] = 1
        
        for i in range(m):
            for j in range(n):
                if i == 0 and j == 0:
                    continue
                current_val = grid[i][j]
                temp = defaultdict(int)
                
                # Check the cell above (i-1, j)
                if i > 0:
                    for xor_val, count in dp[i-1][j].items():
                        new_xor = xor_val ^ current_val
                        temp[new_xor] = (temp[new_xor] + count) % MOD
                
                # Check the cell to the left (i, j-1)
                if j > 0:
                    for xor_val, count in dp[i][j-1].items():
                        new_xor = xor_val ^ current_val
                        temp[new_xor] = (temp[new_xor] + count) % MOD
                
                # Assign the computed values to dp[i][j]
                dp[i][j] = temp
        
        # The result is the count of paths with XOR value k at the bottom-right cell
        return dp[m-1][n-1].get(k, 0) % MOD