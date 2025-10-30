class Solution:
    def minimumOperations(self, nums: List[int]) -> int:
        operations = 0
        while len(set(nums)) != len(nums):
            if not nums:
                break
            operations += 1
            if len(nums) >= 3:
                nums = nums[3:]
            else:
                nums = []
        return operations