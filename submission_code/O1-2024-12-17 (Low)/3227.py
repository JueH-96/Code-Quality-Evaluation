class Solution:
    def findMissingAndRepeatedValues(self, grid: List[List[int]]) -> List[int]:
        n = len(grid)
        freq = [0] * (n*n + 1)  # Using 1-based indexing for convenience
        
        # Count frequency of each number in the grid
        for row in grid:
            for val in row:
                freq[val] += 1
        
        repeated, missing = -1, -1
        # Find the repeated number (freq == 2) and missing number (freq == 0)
        for num in range(1, n*n + 1):
            if freq[num] == 2:
                repeated = num
            elif freq[num] == 0:
                missing = num

        return [repeated, missing]