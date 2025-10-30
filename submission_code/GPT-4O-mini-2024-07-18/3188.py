class Solution:
    def findChampion(self, grid: List[List[int]]) -> int:
        n = len(grid)
        champion = 0
        
        for i in range(1, n):
            if grid[champion][i] == 1:
                continue  # champion is stronger than i
            else:
                champion = i  # i is stronger than champion
        
        return champion