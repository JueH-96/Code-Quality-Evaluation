class Solution:
    def findChampion(self, grid: List[List[int]]) -> int:
        n = len(grid)
        wins = [0] * n
        for i in range(n):
            for j in range(n):
                if grid[i][j] == 1:
                    wins[i] += 1
        for i in range(n):
            if wins[i] == n - 1:
                return i
        return -1