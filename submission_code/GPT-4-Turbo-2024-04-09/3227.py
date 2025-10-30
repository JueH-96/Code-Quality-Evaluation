class Solution:
    def findMissingAndRepeatedValues(self, grid: List[List[int]]) -> List[int]:
        n = len(grid)
        total_elements = n * n
        seen = {}
        repeated = None
        
        # Count occurrences of each number in the grid
        for row in grid:
            for num in row:
                if num in seen:
                    seen[num] += 1
                else:
                    seen[num] = 1
        
        # Find the repeated and the missing number
        for i in range(1, total_elements + 1):
            if i in seen:
                if seen[i] == 2:
                    repeated = i
            else:
                missing = i
        
        return [repeated, missing]