from typing import List

class Solution:
    def findChampion(self, grid: List[List[int]]) -> int:
        n = len(grid)
        for a in range(n):
            champion = True
            for b in range(n):
                if a != b and grid[b][a] == 1:
                    champion = False
                    break
            if champion:
                return a
        # In a valid tournament there must be a champion according to the problem description.
        return -1