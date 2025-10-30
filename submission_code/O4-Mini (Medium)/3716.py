from typing import List

class Solution:
    def longestSubsequence(self, nums: List[int]) -> int:
        # We want the longest subsequence where the absolute differences between
        # consecutive elements form a non-increasing sequence.
        # We'll use a DP that tracks, for each value v (1..300) and each possible
        # last-difference d (0..300), the best length of a subsequence ending at
        # a previous element of value v with last diff ≥ d.
        #
        # best[v][d] = maximum length of a subsequence ending in value v
        #               whose last absolute diff k satisfies k >= d.
        #
        # When we see a new element u = nums[i], for each possible diff d from 0..300
        # we look at the two possible predecessor values:
        #   u-d  (giving diff = d if >=1)
        #   u+d  (giving diff = d if <=300)
        # We take the best[v][d] among those, call it bv.  If bv>0 then we can
        # extend that subsequence by u and get length bv+1 with last diff = d.
        # We store that temporarily in dp[d].  We also seed any first occurrence
        # of a value v to allow forming a pair (length 2) by treating a single
        # element as length 1 seed.  We do that by ensuring best[v][d] >= 1 once v
        # has appeared.
        #
        # After computing dp for u, we update best[u][*] by merging dp[*] and
        # carrying a suffix max (so best[u][d] = max over d..300 of dp[*], old best,
        # and 1 for the seed).
        
        n = len(nums)
        if n == 0:
            return 0
        
        MAXV = 300
        # best[v][d]: suffix-max of lengths for value v with last-diff >= d
        best = [[0] * (MAXV+1) for _ in range(MAXV+1)]
        
        ans = 1  # at least one element
        
        for u in nums:
            # dp[d] = best length of subseq ending at this element u with last-diff exactly d
            dp = [0] * (MAXV+1)
            
            # Compute dp
            for d in range(MAXV+1):
                bv = 0
                v1 = u - d
                if v1 > 0:
                    b = best[v1][d]
                    if b > bv:
                        bv = b
                v2 = u + d
                if v2 <= MAXV:
                    b = best[v2][d]
                    if b > bv:
                        bv = b
                if bv > 0:
                    # We can extend that subsequence by u
                    val = bv + 1
                    dp[d] = val
                    if val > ans:
                        ans = val
            
            # Update best[u][*] with dp[*], old best, and a seed of 1
            bu = best[u]
            suffix_max = 0
            # walk d from MAXV down to 0 to maintain suffix max
            for d in range(MAXV, -1, -1):
                v = bu[d]            # old best
                if dp[d] > v:
                    v = dp[d]
                if suffix_max > v:
                    v = suffix_max
                # seed = 1 to allow forming a pair from a single element
                if v < 1:
                    v = 1
                bu[d] = v
                suffix_max = v
        
        return ans