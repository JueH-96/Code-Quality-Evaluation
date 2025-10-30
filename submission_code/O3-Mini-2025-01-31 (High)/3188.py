from typing import List

class Solution:
    def findChampion(self, grid: List[List[int]]) -> int:
        n = len(grid)
        # A team i is champion if for every other team j,
        # team j is not stronger than team i (i.e., grid[j][i] == 0).
        for i in range(n):
            champion = True
            for j in range(n):
                if i != j and grid[j][i] == 1:  # team j beats team i
                    champion = False
                    break
            if champion:
                return i
        # Given the problem constraints, this line should not be reached.
        return -1