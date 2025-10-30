from typing import List

class Solution:
    def findChampion(self, grid: List[List[int]]) -> int:
        n = len(grid)
        for a in range(n):
            for b in range(n):
                if a != b and grid[b][a]:
                    break
            else:
                return a