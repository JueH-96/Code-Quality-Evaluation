class Solution:
    def findChampion(self, grid: List[List[int]]) -> int:
        n = len(grid)
        for i in range(n):
            if grid[i][i] == 1:
                for j in range(n):
                    if i != j and grid[j][i] == 1:
                        break
                else:
                    return i
        return -1