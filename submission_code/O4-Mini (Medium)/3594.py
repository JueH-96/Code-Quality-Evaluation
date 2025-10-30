from typing import List
from collections import Counter

class Solution:
    def getLargestOutlier(self, nums: List[int]) -> int:
        # Count occurrences and compute total sum
        counts = Counter(nums)
        total_sum = sum(nums)
        
        # Iterate candidate outliers in descending order
        for candidate in sorted(counts.keys(), reverse=True):
            # Let the sum element be S, the outlier be candidate.
            # We need total_sum - candidate - S == S  =>  total_sum - candidate == 2*S
            rem = total_sum - candidate
            if rem & 1:  # must be even to split into 2*S
                continue
            S = rem // 2
            c = counts.get(S, 0)
            # Check if we can pick an index for S distinct from candidate's index
            if S != candidate:
                if c >= 1:
                    return candidate
            else:
                # S == candidate, need at least two occurrences
                if c >= 2:
                    return candidate
        
        # Per problem statement, there is always at least one valid outlier
        # but we add a fallback to satisfy function signature.
        return -1