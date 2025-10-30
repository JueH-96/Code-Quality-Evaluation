class Solution:
    def findMissingAndRepeatedValues(self, grid: List[List[int]]) -> List[int]:
        n = len(grid)
        counts = {}
        for i in range(n):
            for j in range(n):
                num = grid[i][j]
                counts[num] = counts.get(num, 0) + 1
        
        a = -1
        b = -1
        for i in range(1, n * n + 1):
            if i not in counts:
                b = i
            elif counts[i] == 2:
                a = i
        
        return [a, b]