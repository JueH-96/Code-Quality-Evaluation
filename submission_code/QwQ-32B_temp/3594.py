from collections import Counter
from typing import List

class Solution:
    def getLargestOutlier(self, nums: List[int]) -> int:
        total_sum = sum(nums)
        count = Counter(nums)
        max_outlier = -float('inf')
        
        for a in nums:
            b = total_sum - 2 * a
            if b not in count:
                continue
            # Check if a and b can form a valid pair
            if a == b:
                if count[a] < 2:
                    continue
            # If valid, check if this b is a candidate for maximum outlier
            if b > max_outlier:
                max_outlier = b
        
        return max_outlier