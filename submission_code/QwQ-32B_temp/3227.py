from typing import List

class Solution:
    def findMissingAndRepeatedValues(self, grid: List[List[int]]) -> List[int]:
        n = len(grid)
        max_num = n * n
        freq = [0] * (max_num + 1)  # indexes 0 to max_num
        
        for row in grid:
            for num in row:
                freq[num] += 1
        
        a, b = None, None
        for i in range(1, max_num + 1):
            if freq[i] == 2:
                a = i
            elif freq[i] == 0:
                b = i
        
        return [a, b]