class Solution:
    def matrixSum(self, nums: List[List[int]]) -> int:
        # Sort each row in descending order
        for row in nums:
            row.sort(reverse=True)
        
        score = 0
        # Number of columns (elements in each row)
        cols = len(nums[0])
        
        # Process column by column
        for col in range(cols):
            max_in_col = 0
            # Find the maximum element in current column
            for row in range(len(nums)):
                max_in_col = max(max_in_col, nums[row][col])
            score += max_in_col
        
        return score