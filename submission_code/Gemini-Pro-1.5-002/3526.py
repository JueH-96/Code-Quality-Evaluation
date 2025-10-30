class Solution:
    def minFlips(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        def check_rows(g):
            count = 0
            for row in g:
                for i in range(n // 2):
                    if row[i] != row[n - 1 - i]:
                        count += 1
            return count

        def check_cols(g):
            count = 0
            for j in range(n):
                for i in range(m // 2):
                    if g[i][j] != g[m - 1 - i][j]:
                        count += 1
            return count

        q = [(grid, 0)]
        visited = {tuple(tuple(row) for row in grid)}
        while q:
            curr_grid, flips = q.pop(0)
            if check_rows(curr_grid) == 0 or check_cols(curr_grid) == 0:
                return flips
            for i in range(m):
                for j in range(n):
                    next_grid = [list(row) for row in curr_grid]
                    next_grid[i][j] = 1 - next_grid[i][j]
                    grid_tuple = tuple(tuple(row) for row in next_grid)
                    if grid_tuple not in visited:
                        visited.add(grid_tuple)
                        q.append((next_grid, flips + 1))
        return -1