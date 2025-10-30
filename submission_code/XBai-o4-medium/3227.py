from typing import List

class Solution:
    def findMissingAndRepeatedValues(self, grid: List[List[int]]) -> List[int]:
        n = len(grid)
        max_num = n * n
        freq = [0] * (max_num + 1)
        
        for i in range(n):
            for j in range(n):
                num = grid[i][j]
                freq[num] += 1
        
        a, b = 0, 0
        for i in range(1, max_num + 1):
            if freq[i] == 2:
                a = i
            elif freq[i] == 0:
                b = i
        
        return [a, b]