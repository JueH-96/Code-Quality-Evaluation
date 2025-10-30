class Solution:
    def matrixSum(self, nums: List[List[int]]) -> int:
        # Sort each row in descending order
        for row in nums:
            row.sort(reverse=True)
        
        # Initialize score to 0
        score = 0

        # For each column in the matrix
        for col in range(len(nums[0])):
            # Find the maximum number in the column
            max_num = max(row[col] for row in nums)
            # Add the maximum number to the score
            score += max_num

        return score