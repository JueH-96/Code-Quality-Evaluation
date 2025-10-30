class Solution:
    def maxFrequencyScore(self, nums: List[int], k: int) -> int:
        nums.sort()
        l, r, res, val, i = 0, sum(nums) + k * 10 ** 9, 0, 0, 0
        while l < len(nums):
            i = bisect.bisect_right(nums, (r - val + len(nums) - l - 1) // len(nums) - 1) - 1
            if r + k >= nums[i] * (l + 1) + nums[-1] * (len(nums) - l - 1):
                res = max(res, len(nums) - l)
                break
            r -= nums[l]
            val += nums[l]
            l += 1
        return res