class Solution:
    def findMissingAndRepeatedValues(self, grid: List[List[int]]) -> List[int]:
        n = len(grid)
        total_numbers = n * n
        
        # Create a frequency dictionary for numbers in grid
        freq = {}
        for row in grid:
            for num in row:
                freq[num] = freq.get(num, 0) + 1
        
        # Identify the repeated and missing numbers
        repeated = -1
        missing = -1
        
        for num in range(1, total_numbers + 1):
            if num in freq:
                if freq[num] == 2:
                    repeated = num
            else:
                missing = num
        
        return [repeated, missing]