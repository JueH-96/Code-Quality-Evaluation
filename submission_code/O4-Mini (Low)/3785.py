from typing import List

class Solution:
    def countArrays(self, original: List[int], bounds: List[List[int]]) -> int:
        n = len(original)
        
        # Compute prefix deltas: delta[i] = copy[i] - copy[0]
        # For i=0, delta[0]=0; for i>0, delta[i] = sum of original[j]-original[j-1] for j=1..i
        delta = [0] * n
        for i in range(1, n):
            delta[i] = delta[i-1] + (original[i] - original[i-1])
        
        # We want all x = copy[0] such that for every i:
        #   bounds[i][0] <= x + delta[i] <= bounds[i][1]
        # => bounds[i][0] - delta[i] <= x <= bounds[i][1] - delta[i]
        # Intersect all these ranges to get [L, R]
        L, R = -10**30, 10**30  # large initial range
        for i in range(n):
            lo, hi = bounds[i]
            # shifted interval for x
            cur_lo = lo - delta[i]
            cur_hi = hi - delta[i]
            if cur_lo > R or cur_hi < L:
                # no intersection possible
                return 0
            # intersect
            L = max(L, cur_lo)
            R = min(R, cur_hi)
        
        # number of integer x in [L, R]
        if L > R:
            return 0
        return max(0, R - L + 1)