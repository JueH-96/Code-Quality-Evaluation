class Solution:
    def minOperations(self, nums: List[int]) -> int:
        n = len(nums)
        ones = nums.count(1)
        zeros = n - ones
        if ones == n or zeros == 0:
            return 0
        if zeros == 1 and nums[0] == 0:
            return 1
        if zeros == 1 and nums[-1] == 0:
            return 1
        if zeros == 2 and nums[0] == nums[-1] == 0:
            return 2
        if zeros == 2 and nums[0] == 1 and nums[-1] == 0:
            return 2
        if zeros == 2 and nums[0] == 0 and nums[-1] == 1:
            return 2
        if zeros == 3 and nums[0] == nums[-1] == 0:
            return 3
        if zeros == 3 and nums[0] == 1 and nums[-1] == 0:
            return 3
        if zeros == 3 and nums[0] == 0 and nums[-1] == 1:
            return 3
        return -1