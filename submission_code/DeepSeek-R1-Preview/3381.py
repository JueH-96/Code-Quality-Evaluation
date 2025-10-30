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
                    length = j - i + 1
                    if length < min_len:
                        min_len = length
                    break  # No need to check longer subarrays for this i
        return min_len if min_len != float('inf') else -1