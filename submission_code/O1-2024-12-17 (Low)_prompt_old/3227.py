class Solution:
    def findMissingAndRepeatedValues(self, grid: List[List[int]]) -> List[int]:
        n = len(grid)
        size = n * n
        
        # Frequency array to keep count of each number from 1 to n^2
        freq = [0] * (size + 1)
        
        # Populate frequency array
        for row in grid:
            for val in row:
                freq[val] += 1
        
        a = b = -1
        # Identify the repeated and missing values
        for x in range(1, size + 1):
            if freq[x] == 2:
                a = x
            elif freq[x] == 0:
                b = x
        
        return [a, b]