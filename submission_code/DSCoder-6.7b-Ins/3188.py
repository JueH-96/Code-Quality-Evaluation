class Solution:
    def findChampion(self, grid: List[List[int]]) -> int:
        n = len(grid)
        for i in range(n):
            for j in range(n):
                if grid[i][j] == 1:
                    return j
        return -1