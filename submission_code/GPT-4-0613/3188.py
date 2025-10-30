class Solution:
    def findChampion(self, grid: List[List[int]]) -> int:
        n = len(grid)
        for i in range(n):
            if all(grid[i][j] == 1 for j in range(n) if i != j):
                return i
        return -1