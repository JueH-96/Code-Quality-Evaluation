from typing import List

class Solution:
    def findMissingAndRepeatedValues(self, grid: List[List[int]]) -> List[int]:
        n = len(grid)
        total_sum = n * n * (n * n + 1) // 2
        seen = set()
        repeated = None
        
        for row in grid:
            for num in row:
                if num in seen:
                    repeated = num
                seen.add(num)
        
        actual_sum = sum(seen)
        missing = total_sum - actual_sum
        
        return [repeated, missing]