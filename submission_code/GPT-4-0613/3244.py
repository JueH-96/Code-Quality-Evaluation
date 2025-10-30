class Solution:
    def minimumArrayLength(self, nums):
        while len(nums) > 1:
            nums.sort()
            nums.append(nums.pop(0) % nums[0])
        return len(nums)