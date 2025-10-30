import collections
from typing import List

class Solution:
    def getLargestOutlier(self, nums: List[int]) -> int:
        total_sum = sum(nums)
        freq = collections.Counter(nums)
        max_outlier = float('-inf')
        for val in freq:
            if (total_sum - val) % 2 == 0:
                c_val = (total_sum - val) // 2
                if freq[c_val] > 0 and (c_val != val or freq[val] >= 2):
                    if val > max_outlier:
                        max_outlier = max(max_outlier, val)
        return max_outlier