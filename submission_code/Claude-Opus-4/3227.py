class Solution:
    def findMissingAndRepeatedValues(self, grid: List[List[int]]) -> List[int]:
        n = len(grid)
        
        # Flatten the grid and count occurrences
        count = {}
        for row in grid:
            for num in row:
                count[num] = count.get(num, 0) + 1
        
        # Find the repeated number (appears twice)
        repeated = 0
        for num, cnt in count.items():
            if cnt == 2:
                repeated = num
                break
        
        # Find the missing number (not in count)
        missing = 0
        for num in range(1, n * n + 1):
            if num not in count:
                missing = num
                break
        
        return [repeated, missing]