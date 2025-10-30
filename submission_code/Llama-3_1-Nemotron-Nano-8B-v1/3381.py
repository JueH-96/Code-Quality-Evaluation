from typing import List

class Solution:
    def minimumSubarrayLength(self, nums: List[int], k: int) -> int:
        min_length = float('inf')
        n = len(nums)
        for i in range(n):
            current_or = 0
            for j in range(i, n):
                current_or |= nums[j]
                if current_or >= k:
                    current_length = j - i + 1
                    if current_length < min_length:
                        min_length = current_length
                    break  # No need to check longer subarrays for this i
        return min_length if min_length != float('inf') else -1