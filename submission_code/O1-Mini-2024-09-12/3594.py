from typing import List
from collections import Counter

class Solution:
    def getLargestOutlier(self, nums: List[int]) -> int:
        freq = Counter(nums)
        total_sum = sum(nums)
        max_outlier = -float('inf')
        
        for x in freq:
            y = total_sum - 2 * x
            if y in freq:
                if x != y or freq[x] > 1:
                    if y > max_outlier:
                        max_outlier = y
                        
        return max_outlier