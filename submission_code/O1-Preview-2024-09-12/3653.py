class Solution:
    def maxSubarraySum(self, nums: List[int], k: int) -> int:
        n = len(nums)
        s = 0
        max_sum = float('-inf')
        min_prefix_sum = [float('inf')] * k
        min_prefix_sum[0] = 0
        for i in range(n):
            s += nums[i]
            rem = (i + 1) % k
            if min_prefix_sum[rem] != float('inf'):
                max_sum = max(max_sum, s - min_prefix_sum[rem])
            min_prefix_sum[rem] = min(min_prefix_sum[rem], s)
        return max_sum