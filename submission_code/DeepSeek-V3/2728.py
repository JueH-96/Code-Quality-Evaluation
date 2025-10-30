class Solution:
    def matrixSum(self, nums: List[List[int]]) -> int:
        # Sort each row in descending order
        for row in nums:
            row.sort(reverse=True)
        
        score = 0
        # Iterate over each column
        for i in range(len(nums[0])):
            # Find the maximum in the current column across all rows
            max_val = 0
            for row in nums:
                if i < len(row):
                    if row[i] > max_val:
                        max_val = row[i]
            score += max_val
        
        return score