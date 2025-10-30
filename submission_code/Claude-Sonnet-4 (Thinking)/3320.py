class Solution:
    def maxOperations(self, nums: List[int]) -> int:
        if len(nums) < 2:
            return 0
        
        # Perform first operation to get target score
        target_score = nums[0] + nums[1]
        operations = 1
        i = 2  # Start from index 2 (after first two elements)
        
        # Continue while we have pairs and score matches
        while i + 1 < len(nums):
            current_score = nums[i] + nums[i + 1]
            if current_score == target_score:
                operations += 1
                i += 2  # Move to next pair
            else:
                break  # Score doesn't match, stop
        
        return operations