class Solution:
    def minOperations(self, nums: List[int]) -> int:
        operations = 0
        while len(nums) > 1:
            i = 0
            while i < len(nums) - 1:
                if nums[i] == nums[i + 1]:
                    del nums[i]
                    del nums[i]
                    operations += 1
                else:
                    i += 1
            if len(nums) == 1:
                return operations