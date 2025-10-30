class Solution:
    def maxOperations(self, nums: List[int]) -> int:
        if len(nums) < 2:
            return 0
        
        target_score = nums[0] + nums[1]
        operations = 1
        i = 2
        
        while i + 1 < len(nums):
            current_score = nums[i] + nums[i + 1]
            if current_score == target_score:
                operations += 1
                i += 2
            else:
                break
        
        return operations