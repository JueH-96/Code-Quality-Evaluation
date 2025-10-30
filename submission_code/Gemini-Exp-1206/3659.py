class Solution:
    def countPathsWithXorValue(self, grid: List[List[int]], k: int) -> int:
        m, n = len(grid), len(grid[0])
        MOD = 10**9 + 7
        dp = {}

        def solve(row, col, xor_sum):
            if row == m - 1 and col == n - 1:
                if (xor_sum ^ grid[row][col]) == k:
                    return 1
                else:
                    return 0

            if (row, col, xor_sum) in dp:
                return dp[(row, col, xor_sum)]

            ans = 0
            if row + 1 < m:
                ans = (ans + solve(row + 1, col, xor_sum ^ grid[row][col])) % MOD
            if col + 1 < n:
                ans = (ans + solve(row, col + 1, xor_sum ^ grid[row][col])) % MOD

            dp[(row, col, xor_sum)] = ans
            return ans

        return solve(0, 0, 0)