from typing import List

class Solution:
    def longestSubsequence(self, nums: List[int]) -> int:
        n = len(nums)
        # We use dp[v][d] to denote the maximum length of a valid subsequence 
        # ending with the number v (v from 1 to 300) where the “last difference”
        # (i.e. the absolute difference from the previous element) is exactly d.
        # For a subsequence that consists of a single element, we have no difference,
        # so we encode that state as “unrestricted” using d = 301 (a value > any possible diff).
        # When we extend a subsequence ending with some value v with allowed last-diff state,
        # and add a new element x, the new difference becomes d = |x - v|. The subsequence
        # is allowed only if d <= (the last difference that v carried).
        #
        # Because numbers in nums are only in [1,300] and possible differences are 0..300,
        # we allocate dp[1..300][0..301]. We'll use indices 1..300 for values (index 0 unused)
        # and 0..301 for the difference state.
        #
        # Additionally, for efficient queries, for each value v we maintain a "suffix max" array suf[v]
        # defined as: suf[v][d] = max(dp[v][d], dp[v][d+1], ... , dp[v][301]).
        # Then when we want to extend a sequence ending in v with a candidate new difference diff,
        # we can retrieve in O(1) the best chain ending with v that has last diff >= diff.
        
        dp = [[0] * 302 for _ in range(302)]  # dp[v][d] for v in 1..300, d in 0..301
        suf = [[0] * 302 for _ in range(302)]  # suffix maximum arrays for each value v.
        encountered = [False] * 302  # to mark which numbers (1..300) have appeared so far.
        ans = 0

        # Process the numbers in order – this respects the subsequence order.
        for x in nums:
            # new_candidates will hold the best chain lengths we can achieve ending at x
            # in a given state (i.e. with new "last difference" d).
            new_candidates = [0] * 302
            # "Starting new" subsequence with x gives length 1 and state d = 301 (no restriction yet).
            new_candidates[301] = 1
            # Now try to extend any previously processed state.
            # For every value v that has appeared before we can possibly extend a chain ending with v.
            for v in range(1, 301):
                if not encountered[v]:
                    continue
                # Compute the absolute difference for transition from v to x.
                d = x - v if x >= v else v - x
                # For a subsequence ending at v with last difference r (for any r >= d),
                # the best length we can use is suf[v][d]. Then we can extend that chain by adding x.
                cand = suf[v][d] + 1
                if cand > new_candidates[d]:
                    new_candidates[d] = cand

            # Now update our dp state for value x.
            updated = False
            for d in range(302):
                if new_candidates[d] > dp[x][d]:
                    dp[x][d] = new_candidates[d]
                    if dp[x][d] > ans:
                        ans = dp[x][d]
                    updated = True

            # If we’ve improved the dp state for value x, update its suffix maximum array.
            if updated:
                # For brevity, alias dp[x] and suf[x]
                dp_x = dp[x]
                suf_x = suf[x]
                suf_x[301] = dp_x[301]
                for r in range(300, -1, -1):
                    # suf[x][r] is the max of dp[x][r] and suf[x][r+1]
                    temp = dp_x[r]
                    if suf_x[r+1] > temp:
                        temp = suf_x[r+1]
                    suf_x[r] = temp

            # Mark that this number has now been encountered.
            encountered[x] = True

        return ans