from typing import List

class Solution:
    def getLargestOutlier(self, nums: List[int]) -> int:
        total_sum = sum(nums)
        elements = set(nums)
        possible_outliers = []
        for s in nums:
            o = total_sum - 2 * s
            if o in elements:
                possible_outliers.append(o)
        return max(possible_outliers)