from collections import Counter
from typing import List

class Solution:
    def getLargestOutlier(self, nums: List[int]) -> int:
        total = sum(nums)
        freq = Counter(nums)
        max_outlier = -float('inf')
        for s in nums:
            o_candidate = total - 2 * s
            if o_candidate in freq:
                if o_candidate == s:
                    if freq[s] >= 2:
                        if o_candidate > max_outlier:
                            max_outlier = o_candidate
                else:
                    if o_candidate > max_outlier:
                        max_outlier = o_candidate
        return max_outlier