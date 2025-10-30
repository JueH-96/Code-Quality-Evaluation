from typing import List

class Solution:
    def minZeroArray(self, nums: List[int], queries: List[List[int]]) -> int:
        current = nums.copy()
        n = len(current)
        for k in range(len(queries)):
            l, r, val = queries[k]
            # Apply the current query
            for i in range(l, r + 1):
                if current[i] > 0 and current[i] >= val:
                    current[i] -= val
            # Check if all elements are zero
            if all(x == 0 for x in current):
                return k + 1
        return -1