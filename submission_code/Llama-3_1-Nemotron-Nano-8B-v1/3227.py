from typing import List
from collections import Counter

class Solution:
    def findMissingAndRepeatedValues(self, grid: List[List[int]]) -> List[int]:
        n = len(grid)
        expected = n * n
        flattened = [num for row in grid for num in row]
        counts = Counter(flattened)
        
        a = None
        b = None
        
        for num in range(1, expected + 1):
            if counts[num] == 2:
                a = num
            elif counts[num] == 0:
                b = num
        
        return [a, b]