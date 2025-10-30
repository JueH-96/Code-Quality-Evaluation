class Solution:
    def countPathsWithXorValue(self, grid: List[List[int]], k: int) -> int:
        m = len(grid)
        n = len(grid[0])
        mod = 10**9 + 7

        dp = {}  # (row, col, current_xor) -> count

        def solve(r, c, current_xor):
            if (r, c, current_xor) in dp:
                return dp[(r, c, current_xor)]

            if r == m - 1 and c == n - 1:
                return 1 if current_xor == k else 0

            count = 0
            # Move down
            if r + 1 < m:
                count = (count + solve(r + 1, c, current_xor ^ grid[r + 1][c])) % mod

            # Move right
            if c + 1 < n:
                count = (count + solve(r, c + 1, current_xor ^ grid[r][c + 1])) % mod

            dp[(r, c, current_xor)] = count
            return count

        initial_xor = grid[0][0]
        return solve(0, 0, initial_xor)