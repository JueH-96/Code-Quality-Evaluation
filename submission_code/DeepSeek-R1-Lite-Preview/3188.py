from typing import List

class Solution:
    def findChampion(self, grid: List[List[int]]) -> int:
        n = len(grid)
        for a in range(n):
            is_champion = True
            for b in range(n):
                if a != b and grid[b][a] == 1:
                    is_champion = False
                    break
            if is_champion:
                return a
        return -1  # As per problem constraints, there should be a champion.