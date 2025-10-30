class Solution:
    def findChampion(self, grid: List[List[int]]) -> int:
        n = len(grid)
        for i in range(n):
            if all(grid[j][i] == 0 for j in range(n)):
                return i