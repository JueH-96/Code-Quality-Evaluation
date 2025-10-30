class Solution:
    def longestSubsequence(self, nums: List[int]) -> int:
        """
        We want the longest subsequence of nums whose consecutive absolute differences
        form a non-increasing sequence. That is, if the subsequence is [s0, s1, s2, ..., sk],
        then |s1 - s0| >= |s2 - s1| >= ... >= |sk - s(k-1)|.

        A classic way to handle "longest sequence with difference constraints" is dynamic
        programming over (index, last_diff). However, a key observation is that all values
        are in [1..300], so differences are in [0..299]. We can store a DP table:

            dp[i][d] = length of the longest valid subsequence ending at index i
                       with the last absolute difference = d

        Then, to transition:
         - For each j < i, let d' = |nums[i] - nums[j]|.
         - We want d' <= the previous difference in dp[j], call it x. i.e. x >= d'.
           That means dp[i][d'] = max(dp[i][d'], 1 + (max over x >= d' of dp[j][x])).

        We can precompute a "suffix max array" for dp[j], call it pref[j], where
            pref[j][x] = max(dp[j][y]) for all y >= x
        so we can get that maximum in O(1):  pref[j][d']  gives  max_{x >= d'} dp[j][x].

        The overall DP then goes as follows:
         - dp[i][d] is at least 1 (a subsequence of length 1 using just nums[i]).
         - For j in [0..i-1], d' = abs(nums[i] - nums[j]),
           dp[i][d'] = max(dp[i][d'], 1 + pref[j][d']).

        We'll maintain and build pref[i] after we've computed dp[i].

        This is an O(n^2) approach; with n up to 10^4, it can be borderline heavy in Python,
        but careful implementation can still pass under typical time limits.
        """

        n = len(nums)
        if n <= 2:
            return n  # with up to 2 elements, the entire array is trivially valid
        
        # Maximum possible absolute difference is 299 (since 1 <= nums[i] <= 300).
        MAX_DIFF = 300  # we'll use 300 to index [0..299]
        
        # dp[i][d] = the length of the longest valid subsequence ending at index i
        #            with last absolute difference = d
        dp = [[1]*MAX_DIFF for _ in range(n)]
        
        # pref[i][d] = max_{x >= d} dp[i][x]
        # We'll build this after dp[i] is finalized, for quick lookups.
        pref = [[1]*MAX_DIFF for _ in range(n)]
        
        # Build pref for i=0 (all 1s, no real update needed).
        for d in range(MAX_DIFF-2, -1, -1):
            pref[0][d] = max(pref[0][d], pref[0][d+1])
        
        best = 1  # Keep track of the overall best length
        
        for i in range(1, n):
            # By default, dp[i][d] starts at 1 (subseq of length 1: just nums[i])
            # Update dp[i] using dp[j], j < i.
            for j in range(i):
                d_cur = abs(nums[i] - nums[j])  # the proposed difference
                # We want x >= d_cur, so we take the suffix max of dp[j] at d_cur:
                cand = 1 + pref[j][d_cur]
                if cand > dp[i][d_cur]:
                    dp[i][d_cur] = cand
            
            # Build pref[i] for quick lookups in further iterations
            pref[i][MAX_DIFF-1] = dp[i][MAX_DIFF-1]
            for d in range(MAX_DIFF-2, -1, -1):
                pref[i][d] = max(dp[i][d], pref[i][d+1])
            
            # Track the best so far
            best = max(best, max(dp[i]))
        
        return best