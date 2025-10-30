from typing import List
from collections import Counter

class Solution:
    def getLargestOutlier(self, nums: List[int]) -> int:
        S = sum(nums)
        freq = Counter(nums)
        max_outlier = float('-inf')
        
        for a in nums:
            b = S - 2 * a
            if b in freq:
                if a != b or (a == b and freq[a] >= 2):
                    if b > max_outlier:
                        max_outlier = b
                        
        return max_outlier