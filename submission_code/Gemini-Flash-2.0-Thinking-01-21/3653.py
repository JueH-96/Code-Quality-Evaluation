class Solution:
    def maxSubarraySum(self, nums: List[int], k: int) -> int:
        n = len(nums)
        prefix_sum = [0] * (n + 1)
        for i in range(n):
            prefix_sum[i + 1] = prefix_sum[i] + nums[i]
        
        max_sum = -float('inf')
        
        for length in range(k, n + 1, k):
            for i in range(n - length + 1):
                current_sum = prefix_sum[i + length] - prefix_sum[i]
                max_sum = max(max_sum, current_sum)
                
        return max_sum