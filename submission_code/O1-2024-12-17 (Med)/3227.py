class Solution:
    def findMissingAndRepeatedValues(self, grid: List[List[int]]) -> List[int]:
        from collections import defaultdict
        
        n = len(grid)
        freq = defaultdict(int)
        
        for row in grid:
            for num in row:
                freq[num] += 1
        
        # Variables to store the repeated value (a) and missing value (b)
        a, b = None, None
        
        for num in range(1, n*n + 1):
            if freq[num] == 2:
                a = num
            elif freq[num] == 0:
                b = num
            
            if a is not None and b is not None:
                break
        
        return [a, b]