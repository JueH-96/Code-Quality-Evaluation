from typing import List

class Solution:
    def maxRemovals(self, source: str, pattern: str, targetIndices: List[int]) -> int:
        n, m = len(source), len(pattern)
        # Mark which positions are in the removable set
        tmap = [0] * n
        for idx in targetIndices:
            tmap[idx] = 1
        K = len(targetIndices)
        
        # We'll do a DP to match `pattern` as a subsequence of `source`,
        # minimizing how many matches fall on removable positions.
        # dp_prev[j] = min removable‐hits to match first i-1 chars of pattern
        #              using first j chars of source.
        # We roll i from 0..m.
        INF = m + 1
        dp_prev = [0] * (n + 1)
        
        for i in range(m):
            dp_curr = [INF] * (n + 1)
            # dp_curr[0] = INF  (cannot match any positive-length prefix of pattern
            #                     with zero source chars)
            for j in range(1, n + 1):
                # Option 1: skip source[j-1]
                dp_curr[j] = dp_curr[j-1]
                # Option 2: match source[j-1] to pattern[i]
                if source[j-1] == pattern[i]:
                    cost = dp_prev[j-1] + tmap[j-1]
                    if cost < dp_curr[j]:
                        dp_curr[j] = cost
            dp_prev = dp_curr
        
        # dp_prev[n] is the minimal number of removable positions
        # we were forced to keep in order to match the pattern.
        min_kept = dp_prev[n]
        # We can remove all other positions in targetIndices
        return max(0, K - min_kept)