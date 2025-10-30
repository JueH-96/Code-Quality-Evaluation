class Solution:
    def maxOperations(self, nums: List[int]) -> int:
        # The score is determined by the first operation
        target_score = nums[0] + nums[1]
        
        operations = 0
        i = 0
        
        # Keep performing operations while we have at least 2 elements
        while i + 1 < len(nums):
            # Check if the current pair has the same score as target
            if nums[i] + nums[i + 1] == target_score:
                operations += 1
                i += 2  # Move to the next pair
            else:
                # If the score is different, we can't continue
                break
        
        return operations