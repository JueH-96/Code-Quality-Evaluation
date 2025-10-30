class Solution:
    def findMissingAndRepeatedValues(self, grid: List[List[int]]) -> List[int]:
        n = len(grid)
        max_num = n * n
        freq = [0] * (max_num + 1)
        
        for row in grid:
            for num in row:
                freq[num] += 1
        
        repeated = -1
        missing = -1
        
        for i in range(1, max_num + 1):
            if freq[i] == 2:
                repeated = i
            elif freq[i] == 0:
                missing = i
        
        return [repeated, missing]