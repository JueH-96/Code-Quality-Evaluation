class Solution:
    def findChampion(self, grid: List[List[int]]) -> int:
        n = len(grid)
        for i in range(n):
            is_champion = True
            for j in range(n):
                if i == j:
                    continue
                if grid[i][j] != 1:
                    is_champion = False
                    break
            if is_champion:
                return i