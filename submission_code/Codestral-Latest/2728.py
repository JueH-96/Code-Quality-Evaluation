class Solution:
    def matrixSum(self, nums: List[List[int]]) -> int:
        # Sort each row in descending order
        for row in nums:
            row.sort(reverse=True)

        score = 0
        # Iterate over the columns
        for col in range(len(nums[0])):
            max_val = 0
            # Find the maximum value in the current column
            for row in nums:
                if col < len(row):
                    max_val = max(max_val, row[col])
            # Add the maximum value to the score
            score += max_val

        return score