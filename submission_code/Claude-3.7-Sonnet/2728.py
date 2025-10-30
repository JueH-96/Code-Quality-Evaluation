class Solution:
    def matrixSum(self, nums: List[List[int]]) -> int:
        # Sort each row in ascending order
        for i in range(len(nums)):
            nums[i].sort()
        
        score = 0
        # Continue until all rows are empty
        while any(row for row in nums):
            max_removed = 0
            for i in range(len(nums)):
                if nums[i]:  # Check if the row is not empty
                    removed = nums[i].pop()  # Remove the largest element (at the end after sorting)
                    max_removed = max(max_removed, removed)
            score += max_removed
        
        return score