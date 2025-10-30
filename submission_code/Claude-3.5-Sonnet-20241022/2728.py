class Solution:
    def matrixSum(self, nums: List[List[int]]) -> int:
        # Sort each row in descending order
        for row in nums:
            row.sort(reverse=True)
            
        score = 0
        # For each column position
        for col in range(len(nums[0])):
            col_max = 0
            # Find max value in current column position across all rows
            for row in range(len(nums)):
                col_max = max(col_max, nums[row][col])
            score += col_max
            
        return score