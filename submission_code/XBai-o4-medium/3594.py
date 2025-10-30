from collections import Counter
from typing import List

class Solution:
    def getLargestOutlier(self, nums: List[int]) -> int:
        total = sum(nums)
        count = Counter(nums)
        max_outlier = -float('inf')
        
        for O in nums:
            remaining = total - O
            if remaining % 2 != 0:
                continue
            s = remaining // 2
            if s not in count:
                continue
            if O == s:
                if count[s] >= 2:
                    max_outlier = max(max_outlier, O)
            else:
                if count[s] >= 1:
                    max_outlier = max(max_outlier, O)
        
        return max_outlier