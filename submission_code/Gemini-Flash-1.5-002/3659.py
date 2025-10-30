class Solution:
    def countPathsWithXorValue(self, grid: List[List[int]], k: int) -> int:
        m, n = len(grid), len(grid[0])
        mod = 10**9 + 7
        dp = {}

        def count_paths(row, col, current_xor):
            if row == m - 1 and col == n - 1:
                return 1 if (current_xor ^ grid[row][col]) == k else 0

            if (row, col, current_xor) in dp:
                return dp[(row, col, current_xor)]

            count = 0
            next_xor = current_xor ^ grid[row][col]

            if row + 1 < m:
                count = (count + count_paths(row + 1, col, next_xor)) % mod
            if col + 1 < n:
                count = (count + count_paths(row, col + 1, next_xor)) % mod

            dp[(row, col, current_xor)] = count
            return count

        return count_paths(0, 0, 0)