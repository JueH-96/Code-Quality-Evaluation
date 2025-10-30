class Solution:
    def findChampion(self, grid: List[List[int]]) -> int:
        rows, cols = len(grid), len(grid[0])
        for i in range(rows):
            if sum(grid[i]) == cols - 1:
                return i