class Solution:
    def findChampion(self, grid: List[List[int]]) -> int:
        n = len(grid)
        candidate = 0
        
        for i in range(1, n):
            if grid[candidate][i] == 1:
                candidate = i
        
        # Verify candidate is the champion
        for i in range(n):
            if i != candidate and (grid[candidate][i] == 0 or grid[i][candidate] == 1):
                return -1  # No champion found
        
        return candidate