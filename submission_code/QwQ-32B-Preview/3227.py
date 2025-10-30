from typing import List

class Solution:
    def findMissingAndRepeatedValues(self, grid: List[List[int]]) -> List[int]:
        n = len(grid)
        n_squared = n * n
        freq = [0] * (n_squared + 1)
        
        # Flatten the grid and count frequencies
        for row in grid:
            for num in row:
                freq[num] += 1
        
        a = None
        b = None
        for i in range(1, n_squared + 1):
            if freq[i] == 2:
                a = i
            elif freq[i] == 0:
                b = i
            if a is not None and b is not None:
                break
        
        return [a, b]