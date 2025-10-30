from collections import Counter
from typing import List

class Solution:
    def getLargestOutlier(self, nums: List[int]) -> int:
        sum_total = sum(nums)
        counts = Counter(nums)
        max_outlier = -float('inf')
        
        for o in counts:
            sum_remaining = sum_total - o
            if sum_remaining % 2 != 0:
                continue
            s = sum_remaining // 2
            if s not in counts:
                continue
            if s != o:
                if o > max_outlier:
                    max_outlier = o
            else:
                if counts[s] >= 2 and o > max_outlier:
                    max_outlier = o
        
        return max_outlier