class Solution:
    def findChampion(self, grid: List[List[int]]) -> int:
        n = len(grid)
        for i in range(n):
            # Check if column i has all zeros (no team is stronger than team i)
            if sum(grid[j][i] for j in range(n)) == 0:
                return i