class Solution:
    def findMissingAndRepeatedValues(self, grid: List[List[int]]) -> List[int]:
        n = len(grid)
        num_count = {}
        
        # Count the occurrences of each number in the grid
        for row in grid:
            for num in row:
                if num in num_count:
                    num_count[num] += 1
                else:
                    num_count[num] = 1
        
        repeated = -1
        missing = -1
        
        # Find the repeated and missing numbers
        for num in range(1, n * n + 1):
            if num in num_count:
                if num_count[num] == 2:
                    repeated = num
            else:
                missing = num
        
        return [repeated, missing]