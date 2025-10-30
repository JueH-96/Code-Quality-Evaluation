from typing import List

class Solution:
    def maximumTotalSum(self, maximumHeight: List[int]) -> int:
        """
        Greedy: sort bounds descending, then build a strictly descending sequence of
        heights, each no larger than its bound and one less than the previous height.
        This yields the largest possible sum of distinct positive heights or –1 if
        impossible.
        """
        # descending order lets every position grab the largest value still possible
        bounds = sorted(maximumHeight, reverse=True)
        
        prev = float('inf')          # last height we placed (start unbounded)
        total = 0
        
        for bound in bounds:
            # highest height we can give here while staying < prev and ≤ bound
            h = min(bound, prev - 1)
            if h <= 0:               # no positive height available – impossible
                return -1
            total += h
            prev = h                 # next height must be smaller than this one
        
        return total