class Solution:
    def countPathsWithXorValue(self, grid: list[list[int]], k: int) -> int:
        m = len(grid)
        n = len(grid[0])
        MOD = 10**9 + 7

        dp = {}

        def solve(row, col, current_xor):
            if (row, col, current_xor) in dp:
                return dp[(row, col, current_xor)]

            if row == m - 1 and col == n - 1:
                if current_xor ^ grid[row][col] == k:
                    return 1
                else:
                    return 0

            count = 0
            current_xor ^= grid[row][col]

            if row + 1 < m:
                count = (count + solve(row + 1, col, current_xor)) % MOD
            if col + 1 < n:
                count = (count + solve(row, col + 1, current_xor)) % MOD
            
            dp[(row, col, current_xor ^ grid[row][col])] = count
            return count

        return solve(0, 0, 0)