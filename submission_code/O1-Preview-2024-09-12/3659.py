class Solution:
    def countPathsWithXorValue(self, grid: List[List[int]], k: int) -> int:
        m = len(grid)
        n = len(grid[0])
        MOD = 10**9 + 7
        # Since grid values and k are between 0 and 15, the possible XOR values are from 0 to 15
        DP = [ [ [0]*16 for _ in range(n) ] for _ in range(m) ]
        DP[0][0][grid[0][0]] = 1  # Initialize the starting cell
        
        for i in range(m):
            for j in range(n):
                if i == 0 and j == 0:
                    continue  # Skip the starting cell as it's already initialized
                for prev_x in range(16):
                    new_x = prev_x ^ grid[i][j]
                    total = 0
                    if i > 0:
                        total = (total + DP[i-1][j][prev_x]) % MOD
                    if j > 0:
                        total = (total + DP[i][j-1][prev_x]) % MOD
                    DP[i][j][new_x] = (DP[i][j][new_x] + total) % MOD
        return DP[m-1][n-1][k]