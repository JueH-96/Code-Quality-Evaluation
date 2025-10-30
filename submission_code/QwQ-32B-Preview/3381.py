from typing import List

class Solution:
    def minimumSubarrayLength(self, nums: List[int], k: int) -> int:
        n = len(nums)
        min_length = float('inf')
        
        for i in range(n):
            current_OR = 0
            for j in range(i, n):
                current_OR |= nums[j]
                if current_OR >= k:
                    window_length = j - i + 1
                    if window_length < min_length:
                        min_length = window_length
                    break  # No need to extend further for this i
        return min_length if min_length != float('inf') else -1