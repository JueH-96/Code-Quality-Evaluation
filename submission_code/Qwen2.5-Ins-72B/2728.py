from typing import List

class Solution:
    def matrixSum(self, nums: List[List[int]]) -> int:
        score = 0
        while nums[0]:  # Continue until the first row is empty
            max_vals = [max(row) for row in nums]  # Find the max value in each row
            max_val = max(max_vals)  # Find the max value among all max values
            score += max_val  # Add the max value to the score
            for row in nums:
                row.remove(max_val)  # Remove the max value from each row
        return score