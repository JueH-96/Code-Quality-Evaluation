from typing import List

class Solution:
    def maxSubarraySum(self, nums: List[int], k: int) -> int:
        prefix = 0
        min_prefix = [float('inf')] * k
        min_prefix[0] = 0
        max_sum = -float('inf')
        
        for i, num in enumerate(nums):
            prefix += num
            mod = (i + 1) % k
            if min_prefix[mod] != float('inf'):
                current_sum = prefix - min_prefix[mod]
                if current_sum > max_sum:
                    max_sum = current_sum
            # Update the minimum prefix sum for this modulo
            if prefix < min_prefix[mod]:
                min_prefix[mod] = prefix
        
        return max_sum