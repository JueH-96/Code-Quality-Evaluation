class Solution:
    def minOrAfterOperations(self, nums: List[int], k: int) -> int:
        n = len(nums)
        B = 30  # we work with bits 0..29 since 0 <= nums[i] < 2^30

        # Precompute for each bit b an array nxt[b] of length n+1 such that:
        #    nxt[b][i] is the smallest index >= i with (nums[index] >> b) & 1 == 0,
        # or n if none exists.
        nxt = [[n] * (n + 1) for _ in range(B)]
        for b in range(B):
            nxt[b][n] = n
            for i in range(n - 1, -1, -1):
                if (nums[i] >> b) & 1 == 0:
                    nxt[b][i] = i
                else:
                    nxt[b][i] = nxt[b][i + 1]
        
        # Given a candidate X (where we want bits b with X[b]==0 eliminated in the final OR),
        # we want to know if we can partition nums (by merge‐operations) into segments so that
        # for every segment and for every bit b with X[b]==0, the segment contains at least one element 
        # having that bit 0. (That is equivalent to saying the AND of the segment has no 1’s where X has 0.)
        # Because performing k operations leads to (n–k) segments,
        # we require that the maximum number of segments we can achieve (by choosing valid splits)
        # is at least (n – k).
        def feasible(X: int) -> bool:
            forbidden = []
            for b in range(B):
                if ((X >> b) & 1) == 0:
                    forbidden.append(b)
            # If no forbidden bits then there is no constraint; we can split arbitrarily
            if not forbidden:
                return True
            
            # For speed we obtain a list of the "nxt arrays" restricted to the forbidden bits.
            forb_nxt = [nxt[b] for b in forbidden]
            INF = -10**9
            dp = [0] * (n + 1)      # dp[i] = maximum number of valid segments from i to end.
            suffix = [0] * (n + 1)  # suffix[i] = max(dp[i], dp[i+1], …, dp[n])
            dp[n] = 0
            suffix[n] = 0
            
            # We can quickly compute for each starting index i the minimum valid endpoint.
            # For a segment starting at index i, to force for every forbidden bit b that
            # the segment contains a 0 in b, we must extend at least to L = max( forb_nxt[j][i] for j in forbidden ).
            for i in range(n - 1, -1, -1):
                # Use built–in max over the precomputed arrays for the forbidden bits.
                L = max(arr[i] for arr in forb_nxt)
                if L >= n:
                    dp[i] = INF  # no valid segment can start at i
                else:
                    dp[i] = 1 + suffix[L + 1]
                # Update suffix for the DP (this “suffix max” is computed in O(1) per position).
                if dp[i] > suffix[i + 1]:
                    suffix[i] = dp[i]
                else:
                    suffix[i] = suffix[i + 1]
            # To use at most k operations we must have at least (n - k) segments.
            return dp[0] >= n - k
        
        # Now use a greedy “bit–DP”:
        # Start with candidate = (all bits = 1). Then try from highest bit down,
        # if you can “force” that bit off (i.e. replace candidate with candidate with that bit cleared)
        # while still being able to segment the array, then do so.
        candidate = (1 << B) - 1
        for b in range(B - 1, -1, -1):
            test = candidate & ~(1 << b)
            if feasible(test):
                candidate = test
        return candidate