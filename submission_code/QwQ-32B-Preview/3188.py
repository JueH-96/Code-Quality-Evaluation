class Solution:
    def findChampion(self, grid: List[List[int]]) -> int:
        n = len(grid)
        champion_candidate = 0
        for i in range(1, n):
            if grid[i][champion_candidate] == 1:
                champion_candidate = i
        return champion_candidate