class Solution:
    def countPathsWithXorValue(self, grid: List[List[int]], k: int) -> int:
        m, n = len(grid), len(grid[0])
        MOD = 10**9 + 7
        dp = {}

        def solve(r, c, current_xor):
            if (r, c, current_xor) in dp:
                return dp[(r, c, current_xor)]

            if r == m - 1 and c == n - 1:
                return 1 if (current_xor ^ grid[r][c]) == k else 0

            if r >= m or c >= n:
                return 0
            
            ans = 0
            if r + 1 < m:
                ans = (ans + solve(r + 1, c, current_xor ^ grid[r][c])) % MOD
            if c + 1 < n:
                ans = (ans + solve(r, c + 1, current_xor ^ grid[r][c])) % MOD
            
            dp[(r, c, current_xor)] = ans
            return ans

        return solve(0, 0, 0)