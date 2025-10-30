class Solution:
    def findMissingAndRepeatedValues(self, grid: List[List[int]]) -> List[int]:
        n = len(grid)
        counts = {}
        for row in grid:
            for num in row:
                counts[num] = counts.get(num, 0) + 1
        
        repeated = 0
        missing = 0
        for i in range(1, n * n + 1):
            if i not in counts:
                missing = i
            elif counts[i] == 2:
                repeated = i
        
        return [repeated, missing]