from typing import List

class Solution:
    def minimumSumSubarray(self, nums: List[int], l: int, r: int) -> int:
        n = len(nums)
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + nums[i]
        
        min_sum = float('inf')
        
        for current_len in range(l, r + 1):
            max_start = n - current_len
            for i in range(max_start + 1):
                current_sum = prefix[i + current_len] - prefix[i]
                if current_sum > 0:
                    if current_sum < min_sum:
                        min_sum = current_sum
        
        return min_sum if min_sum != float('inf') else -1