from collections import defaultdict
from typing import List

class Solution:
    def longestEqualSubarray(self, nums: List[int], k: int) -> int:
        # Map each value to the list of its indices in nums
        idxs = defaultdict(list)
        for i, x in enumerate(nums):
            idxs[x].append(i)
        
        res = 0
        # For each value, use a sliding window on its occurrence indices
        for positions in idxs.values():
            l = 0
            # r is the right end of the window over the positions list
            for r in range(len(positions)):
                # Compute number of non-x elements in the span [positions[l], positions[r]]
                # span length = positions[r] - positions[l] + 1
                # count of x = (r - l + 1)
                # so non-x = span length - count of x = positions[r] - positions[l] - (r - l)
                while positions[r] - positions[l] - (r - l) > k:
                    l += 1
                # (r - l + 1) is the count of x's we can keep in this window
                res = max(res, r - l + 1)
        
        return res