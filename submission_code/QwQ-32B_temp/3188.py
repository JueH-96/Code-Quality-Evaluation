from typing import List

class Solution:
    def findChampion(self, grid: List[List[int]]) -> int:
        n = len(grid)
        champion = 0
        for j in range(1, n):
            if grid[champion][j] == 0:
                champion = j
        return champion