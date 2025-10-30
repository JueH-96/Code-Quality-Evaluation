class Solution:
    def maxOperations(self, nums: List[int]) -> int:
        n = len(nums)
        if n < 2:
            return 0
        
        target = nums[0] + nums[1]
        operations = 1
        i = 2
        
        while i < n - 1:
            current_sum = nums[i] + nums[i+1]
            if current_sum == target:
                operations += 1
                i += 2
            else:
                break
                
        return operations