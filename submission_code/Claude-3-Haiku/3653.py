class Solution:
    def maxSubarraySum(self, nums: List[int], k: int) -> int:
        n = len(nums)
        prefix_sum = [0] * (n + 1)
        for i in range(n):
            prefix_sum[i + 1] = prefix_sum[i] + nums[i]
        
        max_sum = float('-inf')
        for i in range(0, n + 1, k):
            max_sum = max(max_sum, prefix_sum[i + k] - prefix_sum[i])
        
        return max_sum