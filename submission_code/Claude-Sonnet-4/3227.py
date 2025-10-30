class Solution:
    def findMissingAndRepeatedValues(self, grid: List[List[int]]) -> List[int]:
        n = len(grid)
        total_numbers = n * n
        
        # Count frequency of each number in the grid
        count = {}
        for i in range(n):
            for j in range(n):
                num = grid[i][j]
                count[num] = count.get(num, 0) + 1
        
        repeated = -1
        missing = -1
        
        # Find the repeated number (appears twice)
        for num, freq in count.items():
            if freq == 2:
                repeated = num
                break
        
        # Find the missing number (not in count dict)
        for num in range(1, total_numbers + 1):
            if num not in count:
                missing = num
                break
        
        return [repeated, missing]