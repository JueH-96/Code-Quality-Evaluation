class Solution:
    def maxRemovals(self, source: str, pattern: str, targetIndices: List[int]) -> int:
        """
        We want to remove as many indices in targetIndices as possible while keeping 'pattern'
        as a subsequence of 'source'. Equivalently, we can ask: in some valid subsequence
        match of 'pattern' within 'source', how many of those targetIndices are forced to
        participate in matching the pattern? If a target index is ever used by a valid match,
        we cannot remove it. Hence, to maximize removals, we minimize how many targetIndices
        are actually used in matching 'pattern'.

        Steps:
          1) Let T = set(targetIndices) for quick membership checks.
          2) We'll run a dynamic programming (DP) that computes the minimal number of
             targetIndices used so far to match up to each prefix of 'pattern'.
          3) dp[j] = the minimum count of used targetIndices to match pattern[:j].
             We always skip characters if they do not help (or if skipping yields a better DP).
          4) After processing all characters of 'source', dp[len(pattern)] gives the minimal
             number of T-indices needed to match the entire pattern. The answer is then
             len(targetIndices) - dp[len(pattern)], since we can remove all T-indices not used.

        Complexity:
          - We use a 1D DP array of length (m+1), where m = len(pattern).
          - For each of the n = len(source) characters, we potentially update dp for all j in [m-1..0].
          - Overall O(n*m) time, which is feasible for n,m <= 3000.
        """

        # Convert targetIndices into a set for O(1) membership checks
        T = set(targetIndices)
        n, m = len(source), len(pattern)

        # dp[j] = minimal usage of T-indices to match pattern up to index j (i.e., pattern[:j])
        # dp[0] = 0 means 'matching an empty pattern uses 0 from T'
        INF = 10**9
        dp = [0] + [INF] * m

        # Build the DP bottom-up
        for i in range(n):
            # cost = 1 if this index is in T, else 0
            cost = 1 if i in T else 0
            # We go backward through dp so we don't overwrite dp[j] before using it
            for j in range(m - 1, -1, -1):
                if source[i] == pattern[j]:
                    dp[j + 1] = min(dp[j + 1], dp[j] + cost)

        # dp[m] = minimal number of T-indices used to match the entire pattern
        used_in_match = dp[m]  
        # The rest of targetIndices can be removed
        return len(targetIndices) - used_in_match