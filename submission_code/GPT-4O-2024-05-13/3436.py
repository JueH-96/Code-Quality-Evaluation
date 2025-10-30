from typing import List

class Solution:
    def minimumDifference(self, nums: List[int], k: int) -> int:
        n = len(nums)
        min_diff = float('inf')
        
        for i in range(n):
            current_or = 0
            for j in range(i, n):
                current_or |= nums[j]
                min_diff = min(min_diff, abs(k - current_or))
                
                if min_diff == 0:
                    return 0
        
        return min_diff