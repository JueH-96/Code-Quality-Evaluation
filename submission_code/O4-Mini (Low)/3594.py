from typing import List
from collections import Counter

class Solution:
    def getLargestOutlier(self, nums: List[int]) -> int:
        # Total sum of all elements
        T = sum(nums)
        # Frequency count of all values
        cnt = Counter(nums)
        outlier_candidates = []
        
        # For each value y in nums considered as the "sum element",
        # solve for x = T - 2*y.  If x exists in the array at a different index,
        # then x is a valid outlier.
        for y in nums:
            x = T - 2*y
            # We need at least one occurrence of x in nums, excluding
            # the index used for y if x == y.
            needed = 1 + (1 if x == y else 0)
            if cnt.get(x, 0) >= needed:
                outlier_candidates.append(x)
        
        # Return the maximum possible outlier
        return max(outlier_candidates)