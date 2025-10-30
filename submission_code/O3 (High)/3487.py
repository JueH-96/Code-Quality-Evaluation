from typing import List

class Solution:
    def maxRemovals(self, source: str, pattern: str, targetIndices: List[int]) -> int:
        """
        The idea:
        ----------------
        • Deleting a set T ⊆ targetIndices is legal iff the pattern is still a subsequence
          of the remaining characters.
        • While deletions are done one-by-one, if the pattern is a subsequence after *all*
          deletions in T, it is automatically a subsequence after every intermediate step
          (intermediate strings only contain more characters).  
          → A set of deletions is valid  ⇔  pattern subsequence after the final deletions.
        
        • Therefore we want the largest deletable set T.  
          Equivalently, among all embeddings of pattern in source we want one that touches
          the *fewest* indices in targetIndices; every other deletable index can then be
          removed. If that minimum overlap is `keep`, the answer is
                     len(targetIndices) – keep.
        
        • We compute the minimal possible overlap with a dynamic-programming scan that is
          O(n · m) (n, m ≤ 3000  ⇒  at most 9·10⁶ operations, perfectly fine).

        DP details:
        -----------
        dp[j] … minimal number of kept target-indices needed to match the first j
                characters of `pattern` so far (0 ≤ j ≤ m).  
        Initialise dp[0] = 0, others = +∞.
        Scan the source from left to right.  For every position i:
             cost = 1  if  i ∈ targetIndices   else 0
             update j from m-1 down to 0:
                  if source[i] == pattern[j]:
                        dp[j+1] = min(dp[j+1], dp[j] + cost)
        After the scan, dp[m] is the minimal overlap `keep`.

        Result = len(targetIndices) - dp[m].
        """
        n, m = len(source), len(pattern)
        target_set = set(targetIndices)

        INF = n + 1                       # greater than any possible overlap
        dp = [INF] * (m + 1)
        dp[0] = 0

        for idx, ch in enumerate(source):
            cost = 1 if idx in target_set else 0
            # walk pattern backwards so that current character isn’t reused
            for j in range(m - 1, -1, -1):
                if ch == pattern[j] and dp[j] != INF:
                    dp[j + 1] = min(dp[j + 1], dp[j] + cost)

        # dp[m] is guaranteed finite because pattern is a subsequence of source
        min_overlap = dp[m]
        return len(targetIndices) - min_overlap