class Solution:
    def matrixSum(self, nums: List[List[int]]) -> int:
        # Sort each row in descending order
        for row in nums:
            row.sort(reverse=True)
        
        score = 0
        # Iterate through columns
        for col in range(len(nums[0])):
            max_in_col = 0
            # Find the maximum value in the current column
            for row in nums:
                if col < len(row):
                    max_in_col = max(max_in_col, row[col])
            # Add the maximum value to the score
            score += max_in_col
        
        return score