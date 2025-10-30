from typing import List
from collections import Counter

class Solution:
    def getLargestOutlier(self, nums: List[int]) -> int:
        total = sum(nums)
        freq = Counter(nums)
        possible_outliers = []
        
        # iterate over each candidate X for the sum-of-special numbers.
        for x in freq:
            y = total - 2 * x
            # check existence of y in nums
            if y in freq:
                # if x and y are same, need at least two occurrences (as different indices)
                if x == y and freq[x] < 2:
                    continue
                possible_outliers.append(y)
        
        return max(possible_outliers)