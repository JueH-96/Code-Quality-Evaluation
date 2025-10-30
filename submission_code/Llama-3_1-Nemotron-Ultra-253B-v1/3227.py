from typing import List

class Solution:
    def findMissingAndRepeatedValues(self, grid: List[List[int]]) -> List[int]:
        n = len(grid)
        n_squared = n * n
        freq = [0] * (n_squared + 1)
        
        for row in grid:
            for num in row:
                freq[num] += 1
        
        a, b = None, None
        for num in range(1, n_squared + 1):
            if freq[num] == 2:
                a = num
            elif freq[num] == 0:
                b = num
        
        return [a, b]