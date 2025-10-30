from collections import Counter
from typing import List

class Solution:
    def getLargestOutlier(self, nums: List[int]) -> int:
        total = sum(nums)
        cnt = Counter(nums)
        
        # Get unique candidates sorted in descending order.
        # We consider each candidate from the array as a potential outlier.
        candidates = sorted(cnt.keys(), reverse=True)
        
        for x in candidates:
            # The formula: 2 * S = total - outlier => S = (total - x) / 2 must be an integer.
            if (total - x) % 2 != 0:
                continue  # S must be an integer.
            S = (total - x) // 2
            
            # Two cases:
            # If candidate outlier x equals computed S, then we need at least 2 occurrences in nums:
            if x == S:
                if cnt[S] >= 2:
                    return x
            else:
                if cnt[S] >= 1:
                    return x
                    
        # As per the problem constraints, at least one valid outlier exists.
        return None