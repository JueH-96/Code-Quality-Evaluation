class Solution:
    def canMakeSquare(self, grid: List[List[str]]) -> bool:
        n = 3

        def check(g):
            for i in range(n - 1):
                for j in range(n - 1):
                    if g[i][j] == g[i + 1][j] == g[i][j + 1] == g[i + 1][j + 1]:
                        return True
            return False

        if check(grid):
            return True

        for i in range(n):
            for j in range(n):
                original = grid[i][j]
                grid[i][j] = 'W' if original == 'B' else 'B'
                if check(grid):
                    return True
                grid[i][j] = original

        return False