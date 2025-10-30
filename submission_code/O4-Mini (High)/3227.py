from typing import List

class Solution:
    def findMissingAndRepeatedValues(self, grid: List[List[int]]) -> List[int]:
        n = len(grid)
        size = n * n
        # count occurrences of each number from 1..n^2
        counts = [0] * (size + 1)
        for row in grid:
            for x in row:
                counts[x] += 1
        
        a = b = -1
        # find the one that appears twice (a) and the one that doesn't appear (b)
        for num in range(1, size + 1):
            if counts[num] == 2:
                a = num
            elif counts[num] == 0:
                b = num
        
        return [a, b]