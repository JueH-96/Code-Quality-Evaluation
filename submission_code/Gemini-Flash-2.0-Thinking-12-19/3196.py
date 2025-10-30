class Solution:
    def maxFrequencyScore(self, nums: List[int], k: int) -> int:
        nums.sort()
        n = len(nums)
        prefix_sum = [0] * (n + 1)
        for i in range(n):
            prefix_sum[i + 1] = prefix_sum[i] + nums[i]

        max_freq = 0
        for right in range(n):
            for left in range(right, -1, -1):
                cost = 0
                if left < right:
                    cost = (right - left) * nums[right] - (prefix_sum[right] - prefix_sum[left])
                if cost <= k:
                    max_freq = max(max_freq, right - left + 1)
        return max_freq