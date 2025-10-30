class Solution:
    def maxArrayValue(self, nums: List[int]) -> int:
        n = len(nums)
        res = nums[n - 1]
        for i in range(n - 2, -1, -1):
            if nums[i] <= res:
                res += nums[i]
            else:
                res = nums[i]
        return res