class Solution:
    def missingInteger(self, nums: List[int]) -> int:
        s = nums[0]
        i = 1
        n = len(nums)
        while i < n and nums[i] == nums[i-1] + 1:
            s += nums[i]
            i += 1
        num_set = set(nums)
        x = s
        while x in num_set:
            x += 1
        return x