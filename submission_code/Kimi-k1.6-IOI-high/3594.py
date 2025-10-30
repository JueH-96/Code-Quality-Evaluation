from typing import List
from collections import Counter

class Solution:
    def getLargestOutlier(self, nums: List[int]) -> int:
        total = sum(nums)
        freq = Counter(nums)
        num_set = set(nums)
        max_outlier = -float('inf')
        
        for s in nums:
            o = total - 2 * s
            if o in num_set:
                if s == o:
                    if freq[s] >= 2:
                        if o > max_outlier:
                            max_outlier = o
                else:
                    if o > max_outlier:
                        max_outlier = o
        
        return max_outlier