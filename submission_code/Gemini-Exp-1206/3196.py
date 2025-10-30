class Solution:
    def maxFrequencyScore(self, nums: List[int], k: int) -> int:
        nums.sort()
        n = len(nums)
        left = 0
        curr_sum = 0
        ans = 0
        for right in range(n):
            curr_sum += nums[right]
            while (nums[right] * (right - left + 1) - curr_sum) > k:
                curr_sum -= nums[left]
                left += 1
            ans = max(ans, right - left + 1)
        return ans