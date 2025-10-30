from typing import List

class Solution:
    def minimumSubarrayLength(self, nums: List[int], k: int) -> int:
        n = len(nums)
        res = float('inf')
        for i in range(n):
            curr_or = 0
            for j in range(i, n):
                curr_or |= nums[j]
                if curr_or >= k:
                    res = min(res, j - i + 1)
                    break  # Found the shortest for starting index i
        return res if res != float('inf') else -1