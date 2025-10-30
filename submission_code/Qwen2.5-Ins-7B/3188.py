class Solution:
    def findChampion(self, grid: List[List[int]]) -> int:
        champion = -1
        for i in range(len(grid)):
            if all(grid[i][j] == 1 for j in range(len(grid)) if j != i):
                if champion != -1:
                    return -1  # More than one champion found
                champion = i
        return champion