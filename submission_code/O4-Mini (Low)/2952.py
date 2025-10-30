from typing import List
import bisect

class Solution:
    def minimumTime(self, nums1: List[int], nums2: List[int], x: int) -> int:
        n = len(nums1)
        # Pair slopes with initial values, sort slopes descending
        pairs = sorted(zip(nums2, nums1), key=lambda p: p[0], reverse=True)
        slopes = [p[0] for p in pairs]
        initials = [p[1] for p in pairs]
        # Prefix sums of slopes and initials
        pref_s = [0]*(n+1)
        pref_i = [0]*(n+1)
        for i in range(n):
            pref_s[i+1] = pref_s[i] + slopes[i]
            pref_i[i+1] = pref_i[i] + initials[i]
        total_initial = sum(nums1)
        total_slopes = pref_s[n]
        
        # Compute S(t): minimum possible total after t seconds
        def S(t: int) -> int:
            # base sum after increments, before resets
            base = total_initial + t * total_slopes
            # we can reset at most min(t, n) distinct indices
            k = min(t, n)
            # savings = sum of initial values of those k + sum over j=1..k of r_j * slope_j
            # where r_j = t - j + 1, slopes_j are top k slopes
            sum_init = pref_i[k]
            # compute weighted sum of top k slopes
            # sum_{j=1..k}(t-j+1)*slopes[j-1]
            # = t*(sum slopes[0..k-1]) - sum_{j=1..k} (j-1)*slopes[j-1]
            sum_sl = pref_s[k]
            weighted = t * sum_sl
            # subtract sum (j-1)*slopes[j-1]
            acc = 0
            for j in range(k):
                acc += j * slopes[j]
            weighted -= acc
            savings = sum_init + weighted
            return base - savings
        
        # If even at t = n we can't reach <= x, further t won't help
        if S(n) > x:
            return -1
        # Binary search for minimal t in [0..n]
        lo, hi = 0, n
        while lo < hi:
            mid = (lo + hi) // 2
            if S(mid) <= x:
                hi = mid
            else:
                lo = mid + 1
        return lo