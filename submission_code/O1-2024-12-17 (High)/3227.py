class Solution:
    def findMissingAndRepeatedValues(self, grid: List[List[int]]) -> List[int]:
        n = len(grid)
        freq = [0] * (n*n + 1)
        
        # Count the frequency of each number
        for row in grid:
            for val in row:
                freq[val] += 1
        
        repeated = -1
        missing = -1
        
        # Find the repeated and missing values
        for num in range(1, n*n + 1):
            if freq[num] == 2:
                repeated = num
            elif freq[num] == 0:
                missing = num
        
        return [repeated, missing]