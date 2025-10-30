class Solution:
    def findMissingAndRepeatedValues(self, grid: List[List[int]]) -> List[int]:
        n = len(grid)
        count = [0] * (n * n + 1)  # index 0 won't be used
        
        # Count occurrences of each number in the grid
        for i in range(n):
            for j in range(n):
                count[grid[i][j]] += 1
        
        repeated = missing = 0
        
        # Find the repeated and missing numbers
        for i in range(1, n * n + 1):
            if count[i] == 2:
                repeated = i
            elif count[i] == 0:
                missing = i
        
        return [repeated, missing]