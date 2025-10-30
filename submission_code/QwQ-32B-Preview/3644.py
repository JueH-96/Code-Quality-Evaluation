from typing import List

class Solution:
    def minimumSumSubarray(self, nums: List[int], l: int, r: int) -> int:
        n = len(nums)
        if l > n or r > n:
            return -1
        
        # Build prefix sum array
        prefix = [0] * (n + 1)
        for i in range(1, n + 1):
            prefix[i] = prefix[i - 1] + nums[i - 1]
        
        min_sum = None
        for i in range(n):
            for k in range(l, r + 1):
                if i + k <= n:
                    current_sum = prefix[i + k] - prefix[i]
                    if current_sum > 0:
                        if min_sum is None or current_sum < min_sum:
                            min_sum = current_sum
        
        return min_sum if min_sum is not None else -1