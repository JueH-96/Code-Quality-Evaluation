from typing import List

class Solution:
    def minimumSubarrayLength(self, nums: List[int], k: int) -> int:
        n = len(nums)
        min_len = float('inf')
        
        for i in range(n):
            current_or = 0
            for j in range(i, n):
                current_or |= nums[j]
                if current_or >= k:
                    current_len = j - i + 1
                    if current_len < min_len:
                        min_len = current_len
                    break  # Break early as further elements will only increase the length
        
        return -1 if min_len == float('inf') else min_len