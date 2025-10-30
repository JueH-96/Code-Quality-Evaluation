class Solution:
    def findMissingAndRepeatedValues(self, grid: List[List[int]]) -> List[int]:
        n = len(grid)
        
        # Count frequencies of all numbers in the grid
        freq = {}
        for i in range(n):
            for j in range(n):
                num = grid[i][j]
                freq[num] = freq.get(num, 0) + 1
        
        repeating = missing = None
        
        # Check each number from 1 to n*n
        for i in range(1, n * n + 1):
            if i in freq and freq[i] == 2:
                repeating = i
            elif i not in freq:
                missing = i
        
        return [repeating, missing]