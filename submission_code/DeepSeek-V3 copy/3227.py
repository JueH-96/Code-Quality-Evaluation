from collections import defaultdict

class Solution:
    def findMissingAndRepeatedValues(self, grid: List[List[int]]) -> List[int]:
        n = len(grid)
        total_numbers = n * n
        count = defaultdict(int)
        
        # Flatten the grid and count occurrences
        for row in grid:
            for num in row:
                count[num] += 1
        
        # Find the repeated and missing numbers
        a = None
        b = None
        for num in range(1, total_numbers + 1):
            if count[num] == 2:
                a = num
            elif count[num] == 0:
                b = num
        
        return [a, b]