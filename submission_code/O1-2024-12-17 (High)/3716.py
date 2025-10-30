from typing import List

class Solution:
    def longestSubsequence(self, nums: List[int]) -> int:
        n = len(nums)
        # dp[i][d] = length of the longest valid subsequence ending exactly at index i
        #            where the absolute difference to its previous element is d
        # We use a "sentinel" difference of 300 to represent a subsequence of length 1
        # (i.e., no previous difference has been fixed yet).
        #
        # pref[i][d] = max( dp[i][x] for all x >= d ) 
        #   i.e. the best length ending at i with last difference >= d
        #
        # Recurrence:
        #   For each i, we start dp[i][300] = 1 (single element subsequence).
        #   For each j < i:
        #       d = |nums[i] - nums[j]|
        #       dp[i][d] = max(dp[i][d], pref[j][d] + 1)
        #
        #   Then we build pref[i][..] by scanning downward:
        #       pref[i][300] = dp[i][300]
        #       for d in 299..0:
        #           pref[i][d] = max(pref[i][d+1], dp[i][d], pref[i][300])
        #
        # The answer is the maximum dp[i][d] found for all i and d.

        # dp and pref each have size n x 301 (0..300)
        dp = [[0]*301 for _ in range(n)]
        pref = [[0]*301 for _ in range(n)]
        
        answer = 1
        for i in range(n):
            # Single-element subsequence at i
            dp[i][300] = 1
            if dp[i][300] > answer:
                answer = dp[i][300]
            
            # Update dp[i] by trying to extend subsequences ending at j < i
            dp_i = dp[i]  # local reference to speed up lookups
            for j in range(i):
                d = nums[i] - nums[j]
                if d < 0:
                    d = -d
                # pref[j][d] is the best length that ends at j with difference >= d
                new_val = pref[j][d] + 1
                if new_val > dp_i[d]:
                    dp_i[d] = new_val
                    if new_val > answer:
                        answer = new_val
            
            # Build pref[i] after dp[i] is fully updated
            pref_i = pref[i]
            pref_i[300] = dp_i[300]
            # We scan down from 299 to 0 to fill the prefix maxima
            for d in range(299, -1, -1):
                best_here = dp_i[d]
                if pref_i[d+1] > best_here:
                    best_here = pref_i[d+1]
                if pref_i[300] > best_here:
                    best_here = pref_i[300]
                pref_i[d] = best_here
        
        return answer