class Solution:
    def minOperations(self, nums: List[int]) -> int:
        n = len(nums)
        if n % 2 != 0 and nums[-1] == 0:
            return -1
        
        ops = 0
        
        for i in range(n - 3):
            if nums[i] == 0:
                ops += 1
                nums[i] = 1
                nums[i+1] = 1 - nums[i+1]
                nums[i+2] = 1 - nums[i+2]
        
        if nums[-3] == 0:
            ops += 1
            nums[-3] = 1
            nums[-2] = 1 - nums[-2]
            nums[-1] = 1 - nums[-1]
        
        if nums[-2] == 0 and nums[-1] == 0:
            return -1
        elif nums[-2] == 0:
            ops += 2
        elif nums[-1] == 0:
            ops += 1
        
        return ops