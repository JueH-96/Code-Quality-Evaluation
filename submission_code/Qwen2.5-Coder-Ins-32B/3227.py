from typing import List

class Solution:
    def findMissingAndRepeatedValues(self, grid: List[List[int]]) -> List[int]:
        n = len(grid)
        total_numbers = n * n
        num_count = [0] * (total_numbers + 1)
        repeated = -1
        missing = -1
        
        for row in grid:
            for num in row:
                num_count[num] += 1
        
        for num in range(1, total_numbers + 1):
            if num_count[num] == 2:
                repeated = num
            elif num_count[num] == 0:
                missing = num
        
        return [repeated, missing]