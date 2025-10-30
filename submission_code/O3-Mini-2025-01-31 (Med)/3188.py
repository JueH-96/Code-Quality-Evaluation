from typing import List

class Solution:
    def findChampion(self, grid: List[List[int]]) -> int:
        n = len(grid)
        champion = 0
        for i in range(1, n):
            # if team i is stronger than current champion, update champion
            if grid[i][champion] == 1:
                champion = i
        return champion