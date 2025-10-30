class Solution:
    def countPathsWithXorValue(self, grid: List[List[int]], k: int) -> int:
        m = len(grid)
        n = len(grid[0])
        MOD = 10**9 + 7
        max_xor_val = 1024  # Increased range for XOR values
        dp = [[[0] * max_xor_val for _ in range(n)] for _ in range(m)]

        dp[0][0][grid[0][0]] = 1

        for r in range(m):
            for c in range(n):
                if r == 0 and c == 0:
                    continue
                for xor_val in range(max_xor_val):
                    current_val = grid[r][c]
                    prev_xor_needed = xor_val ^ current_val
                    if r > 0:
                        if 0 <= prev_xor_needed < max_xor_val:
                            dp[r][c][xor_val] = (dp[r][c][xor_val] + dp[r-1][c][prev_xor_needed]) % MOD
                    if c > 0:
                        if 0 <= prev_xor_needed < max_xor_val:
                            dp[r][c][xor_val] = (dp[r][c][xor_val] + dp[r][c-1][prev_xor_needed]) % MOD

        return dp[m-1][n-1][k]