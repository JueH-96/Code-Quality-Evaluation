from typing import List
from collections import Counter

class Solution:
    def getLargestOutlier(self, nums: List[int]) -> int:
        cnt = Counter(nums)
        sum_total = sum(nums)
        max_outlier = -float('inf')
        for s in nums:
            o = sum_total - 2 * s
            if o in cnt:
                if o == s:
                    if cnt[s] >= 2:
                        if o > max_outlier:
                            max_outlier = o
                else:
                    if o > max_outlier:
                        max_outlier = o
        return max_outlier