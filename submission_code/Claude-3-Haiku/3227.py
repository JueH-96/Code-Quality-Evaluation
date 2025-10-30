class Solution:
    def findMissingAndRepeatedValues(self, grid: List[List[int]]) -> List[int]:
        n = len(grid)
        seen = [False] * (n * n + 1)
        
        # Find the repeating number
        for i in range(n):
            for j in range(n):
                num = grid[i][j]
                if seen[num]:
                    repeating = num
                else:
                    seen[num] = True
        
        # Find the missing number
        for i in range(1, n * n + 1):
            if not seen[i]:
                missing = i
                break
        
        return [repeating, missing]