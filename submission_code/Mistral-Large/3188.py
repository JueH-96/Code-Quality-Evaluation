from typing import List

class Solution:
    def findChampion(self, grid: List[List[int]]) -> int:
        n = len(grid)
        champion = 0

        for i in range(1, n):
            is_champion = True
            for j in range(n):
                if grid[j][i] == 1:
                    is_champion = False
                    break
            if is_champion:
                champion = i
                break

        return champion