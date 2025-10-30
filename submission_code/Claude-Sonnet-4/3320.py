class Solution:
    def maxOperations(self, nums: List[int]) -> int:
        if len(nums) < 2:
            return 0
        
        # Establish the target score from the first operation
        target_score = nums[0] + nums[1]
        operations = 1
        
        # Start from index 2 (after removing first two elements)
        i = 2
        
        # Continue while we have at least 2 elements and can match the score
        while i + 1 < len(nums):
            current_score = nums[i] + nums[i + 1]
            if current_score == target_score:
                operations += 1
                i += 2  # Move to next pair
            else:
                break  # Can't continue with same score
        
        return operations