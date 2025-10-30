class Solution:
    def maxOperations(self, nums: List[int]) -> int:
        if len(nums) < 2:
            return 0
        
        target_score = nums[0] + nums[1]
        operations = 1
        
        for i in range(2, len(nums) - 1, 2):
            if nums[i] + nums[i+1] == target_score:
                operations += 1
            else:
                break
        
        return operations