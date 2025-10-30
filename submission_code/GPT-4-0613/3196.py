class Solution:
    def maxFrequencyScore(self, nums, k):
        nums.sort()
        left = 0
        for right in range(len(nums)):
            k += nums[right - 1]
            if nums[right] * (right - left) > k:
                k -= nums[left]
                left += 1
        return right - left + 1