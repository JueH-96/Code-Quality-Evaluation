from typing import List

class Solution:
    def maxSubarraySum(self, nums: List[int], k: int) -> int:
        n = len(nums)
        prefix_sum = [0] * (n + 1)
        for i in range(1, n + 1):
            prefix_sum[i] = prefix_sum[i - 1] + nums[i - 1]
        
        min_prefix = {0: 0}
        max_sum = float('-inf')
        
        for i in range(1, n + 1):
            current_sum = prefix_sum[i]
            mod = i % k
            if mod in min_prefix:
                candidate = current_sum - min_prefix[mod]
                if candidate > max_sum:
                    max_sum = candidate
            
            current_mod = i % k
            if current_mod in min_prefix:
                if current_sum < min_prefix[current_mod]:
                    min_prefix[current_mod] = current_sum
            else:
                min_prefix[current_mod] = current_sum
        
        return max_sum