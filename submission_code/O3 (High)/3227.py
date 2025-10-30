from typing import List

class Solution:
    def findMissingAndRepeatedValues(self, grid: List[List[int]]) -> List[int]:
        n = len(grid)
        total_nums = n * n                     # numbers are in the range [1, n²]
        
        seen = [False] * (total_nums + 1)      # index 0 is unused
        repeated = missing = -1
        
        # Scan the matrix to find the repeated number
        for row in grid:
            for val in row:
                if seen[val]:
                    repeated = val             # this value appears twice
                else:
                    seen[val] = True
        
        # The number that was never seen is the missing one
        for num in range(1, total_nums + 1):
            if not seen[num]:
                missing = num
                break
        
        return [repeated, missing]