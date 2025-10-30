class Solution:
    def lenOfVDiagonal(self, grid: List[List[int]]) -> int:
        n, m = len(grid), len(grid[0])
        max_length = 0

        def dfs(i, j, di, dj, length):
            nonlocal max_length
            if i < 0 or i >= n or j < 0 or j >= m or grid[i][j] != 1:
                return

            max_length = max(max_length, length)
            grid[i][j] = 0  # Mark as visited
            dfs(i + di, j + dj, di, dj, length + 1)
            dfs(i - di, j - dj, -di, -dj, length + 1)

        for i in range(n):
            for j in range(m):
                if grid[i][j] == 1:
                    dfs(i, j, 1, 1, 1)  # Top-left to bottom-right
                    dfs(i, j, 1, -1, 1)  # Top-right to bottom-left
                    dfs(i, j, -1, 1, 1)  # Bottom-left to top-right
                    dfs(i, j, -1, -1, 1)  # Bottom-right to top-left

        return max_length