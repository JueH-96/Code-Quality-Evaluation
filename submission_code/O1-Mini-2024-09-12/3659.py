from typing import List

class Solution:
    def countPathsWithXorValue(self, grid: List[List[int]], k: int) -> int:
        MOD = 10**9 + 7
        m, n = len(grid), len(grid[0])
        
        # Initialize DP array: previous row and current row
        prev_row = [[0]*16 for _ in range(n)]
        
        # Initialize the first cell
        initial_xor = grid[0][0]
        prev_row[0][initial_xor] = 1
        
        # Fill the first row
        for j in range(1, n):
            for x in range(16):
                new_xor = x ^ grid[0][j]
                prev_row[j][new_xor] = prev_row[j-1][x] % MOD
        # Iterate over the rest of the rows
        for i in range(1, m):
            current_row = [[0]*16 for _ in range(n)]
            for j in range(n):
                for prev_xor in range(16):
                    new_xor = prev_xor ^ grid[i][j]
                    if j > 0:
                        current_row[j][new_xor] = (current_row[j][new_xor] + current_row[j-1][prev_xor]) % MOD
                    if i > 0:
                        current_row[j][new_xor] = (current_row[j][new_xor] + prev_row[j][prev_xor]) % MOD
            prev_row = current_row
        
        return prev_row[-1][k] % MOD