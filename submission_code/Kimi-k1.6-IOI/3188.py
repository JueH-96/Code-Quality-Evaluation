from typing import List

class Solution:
    def findChampion(self, grid: List[List[int]]) -> int:
        n = len(grid)
        for a in range(n):
            for b in range(n):
                if b != a and grid[b][a] == 1:
                    break
            else:
                return a
        return -1  # This line is theoretically unreachable due to problem constraints