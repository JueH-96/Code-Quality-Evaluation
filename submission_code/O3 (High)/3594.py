from typing import List
from collections import Counter

class Solution:
    def getLargestOutlier(self, nums: List[int]) -> int:
        # Total sum of the array
        total = sum(nums)
        # Frequency of each value
        freq = Counter(nums)

        # Check possible outlier values from largest to smallest
        for out_val in sorted(freq.keys(), reverse=True):
            remain = total - out_val        # sum of the other n-1 elements
            if remain & 1:                  # must be even so that remain = 2 * (sum element)
                continue
            sum_val = remain // 2           # value that has to appear as the 'sum element'
            if sum_val not in freq:         # such a value does not exist
                continue

            # Need distinct indices for outlier and sum element
            if sum_val == out_val:
                # the same value must occur at least twice
                if freq[out_val] >= 2:
                    return out_val
            else:
                # different values → indices automatically distinct
                return out_val

        # Problem guarantees at least one outlier exists
        # (This line is just a safeguard)
        return 0