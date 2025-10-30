from typing import List

class Solution:
    def findMissingAndRepeatedValues(self, grid: List[List[int]]) -> List[int]:
        n = len(grid)
        total_numbers = n * n
        count = {}
        
        # Flatten the grid and count occurrences
        for row in grid:
            for num in row:
                if num in count:
                    count[num] += 1
                else:
                    count[num] = 1
        
        # Find the repeating number
        a = None
        for num, cnt in count.items():
            if cnt == 2:
                a = num
                break
        
        # Find the missing number
        b = None
        for num in range(1, total_numbers + 1):
            if num not in count:
                b = num
                break
        
        return [a, b]