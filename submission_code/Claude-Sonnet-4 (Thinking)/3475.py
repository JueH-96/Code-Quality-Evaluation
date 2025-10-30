class Solution:
    def minOperations(self, nums: List[int]) -> int:
        n = len(nums)
        operations = 0
        
        for i in range(n):
            if nums[i] == 0:
                if i > n - 3:
                    return -1
                # Flip at position i
                nums[i] = 1 - nums[i]
                nums[i+1] = 1 - nums[i+1]
                nums[i+2] = 1 - nums[i+2]
                operations += 1
        
        return operations