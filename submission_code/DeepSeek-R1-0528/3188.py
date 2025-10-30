class Solution:
    def findChampion(self, grid: List[List[int]]) -> int:
        n = len(grid)
        for i in range(n):
            if all(grid[k][i] == 0 for k in range(n)):
                return i