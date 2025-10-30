from typing import List

class Solution:
    def findChampion(self, grid: List[List[int]]) -> int:
        n = len(grid)
        # Use elimination to find the candidate champion.
        candidate = 0
        for i in range(1, n):
            # If candidate loses to i, then i could be the champion.
            if grid[candidate][i] == 0:
                candidate = i
        # candidate is the one with no stronger team above it
        return candidate