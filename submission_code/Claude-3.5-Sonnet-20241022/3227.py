class Solution:
    def findMissingAndRepeatedValues(self, grid: List[List[int]]) -> List[int]:
        n = len(grid)
        nums = set()
        repeated = 0
        
        # Find repeated number
        for i in range(n):
            for j in range(n):
                if grid[i][j] in nums:
                    repeated = grid[i][j]
                nums.add(grid[i][j])
        
        # Find missing number
        missing = 0
        for i in range(1, n*n + 1):
            if i not in nums:
                missing = i
                break
                
        return [repeated, missing]