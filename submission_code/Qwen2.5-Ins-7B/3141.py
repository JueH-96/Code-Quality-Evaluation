from typing import List

class Solution:
    def minSizeSubarray(self, nums: List[int], target: int) -> int:
        n = len(nums)
        prefix_sum = [0] * (n + 1)
        for i in range(n):
            prefix_sum[i + 1] = prefix_sum[i] + nums[i]
        
        total_sum = prefix_sum[-1]
        if total_sum == 0:
            return target == 0 and n or -1
        
        target %= total_sum
        if target == 0:
            return n * (target // total_sum)
        
        min_len = float('inf')
        left = 0
        for right in range(n):
            while prefix_sum[right + 1] - prefix_sum[left] > target:
                left += 1
            if prefix_sum[right + 1] - prefix_sum[left] == target:
                min_len = min(min_len, right - left + 1)
        
        if min_len == float('inf'):
            return -1
        
        full_copies = target // total_sum
        return min_len + full_copies * n