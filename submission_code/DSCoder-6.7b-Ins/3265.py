class Solution:
    def maximumSubarraySum(self, nums: List[int], k: int) -> int:
        n = len(nums)
        max_sum = 0
        window_sum = sum(nums[:k])
        if abs(nums[0] - nums[n-1]) == k:
            max_sum = window_sum
        for i in range(k, n):
            window_sum = window_sum - nums[i-k] + nums[i]
            if abs(nums[i-k+1] - nums[i]) == k and window_sum > max_sum:
                max_sum = window_sum
        return max_sum