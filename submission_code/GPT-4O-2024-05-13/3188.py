from typing import List

class Solution:
    def findChampion(self, grid: List[List[int]]) -> int:
        n = len(grid)
        champion = 0
        
        for i in range(1, n):
            if grid[i][champion] == 1:
                champion = i
        
        for i in range(n):
            if i != champion and (grid[i][champion] == 1 or grid[champion][i] == 0):
                return -1
        
        return champion