from typing import List

class Solution:
    def maxSubarraySum(self, nums: List[int], k: int) -> int:
        n = len(nums)
        prefix_sum = [0] * (n + 1)
        for i in range(1, n + 1):
            prefix_sum[i] = prefix_sum[i - 1] + nums[i - 1]
        
        min_prefix = [float('inf')] * k
        min_prefix[0] = prefix_sum[0]
        
        max_sum = -float('inf')
        
        for r in range(n):
            rem = (r + 1) % k
            if min_prefix[rem] != float('inf'):
                current_sum = prefix_sum[r + 1] - min_prefix[rem]
                if current_sum > max_sum:
                    max_sum = current_sum
            new_rem = (r + 1) % k
            if prefix_sum[r + 1] < min_prefix[new_rem]:
                min_prefix[new_rem] = prefix_sum[r + 1]
        
        return max_sum