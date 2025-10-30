from typing import List

class Solution:
    def minimumDifference(self, nums: List[int], k: int) -> int:
        n = len(nums)
        min_diff = float('inf')
        current_or = 0
        
        for i in range(n):
            current_or = 0
            for j in range(i, n):
                current_or |= nums[j]
                diff = abs(k - current_or)
                min_diff = min(min_diff, diff)
                if min_diff == 0:
                    return 0
        
        return min_diff