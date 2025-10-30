class Solution:
    def maxOperations(self, nums: List[int]) -> int:
        if len(nums) < 2:
            return 0
        
        operations = 0
        score = nums[0] + nums[1]
        nums = nums[2:]
        operations += 1
        
        while len(nums) >= 2:
            current_score = nums[0] + nums[1]
            if current_score == score:
                nums = nums[2:]
                operations += 1
            else:
                break
                
        return operations