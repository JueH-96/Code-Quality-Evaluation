class Solution:
    def countPathsWithXorValue(self, grid: List[List[int]], k: int) -> int:
        m, n = len(grid), len(grid[0])
        dp = {}

        def solve(i, j, xor):
            if (i, j, xor) in dp:
                return dp[(i, j, xor)]

            if i == m - 1 and j == n - 1:
                if (xor ^ grid[i][j]) == k:
                    return 1
                else:
                    return 0

            count = 0
            current_xor = xor ^ grid[i][j]

            if i + 1 < m:
                count = (count + solve(i + 1, j, current_xor)) % (10**9 + 7)
            if j + 1 < n:
                count = (count + solve(i, j + 1, current_xor)) % (10**9 + 7)

            dp[(i, j, xor)] = count
            return count

        return solve(0, 0, 0)