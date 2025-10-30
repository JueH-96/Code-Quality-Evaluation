class Solution:
    def minOperations(self, nums: List[int]) -> int:
        n = len(nums)
        operations = 0
        i = 0
        
        while i < n:
            if nums[i] == 0:
                if i + 2 >= n:
                    return -1
                operations += 1
                nums[i] = 1
                nums[i + 1] = 1 - nums[i + 1]
                nums[i + 2] = 1 - nums[i + 2]
            i += 1
        
        return operations