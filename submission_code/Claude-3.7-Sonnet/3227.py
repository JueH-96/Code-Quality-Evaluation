class Solution:
    def findMissingAndRepeatedValues(self, grid: List[List[int]]) -> List[int]:
        n = len(grid)
        count = [0] * (n * n + 1)
        
        # Count occurrences of each number
        for i in range(n):
            for j in range(n):
                count[grid[i][j]] += 1
        
        a, b = -1, -1
        
        # Find the repeated and missing numbers
        for i in range(1, n * n + 1):
            if count[i] == 2:
                a = i
            elif count[i] == 0:
                b = i
        
        return [a, b]