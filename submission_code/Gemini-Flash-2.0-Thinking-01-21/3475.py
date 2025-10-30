class Solution:
    def minOperations(self, nums: List[int]) -> int:
        n = len(nums)
        nums = list(nums) # make a mutable copy
        operations = 0
        for i in range(n - 2):
            if nums[i] == 0:
                operations += 1
                for j in range(i, i + 3):
                    nums[j] = 1 - nums[j]
        for i in range(n):
            if nums[i] == 0:
                return -1
        return operations