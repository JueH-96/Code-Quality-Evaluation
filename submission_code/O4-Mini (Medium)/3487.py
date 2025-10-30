from typing import List

class Solution:
    def maxRemovals(self, source: str, pattern: str, targetIndices: List[int]) -> int:
        n = len(source)
        m = len(pattern)
        # mark which positions are candidates for removal
        is_removable = [False] * n
        for idx in targetIndices:
            is_removable[idx] = True
        # dp[j] = minimum number of kept-removals (positions in targetIndices that we must keep)
        # to match first j chars of pattern in the processed prefix of source
        INF = 10**9
        dp_prev = [INF] * (m + 1)
        dp_prev[0] = 0  # to match empty pattern costs 0
        # iterate over source positions
        for i, ch in enumerate(source):
            cost = 1 if is_removable[i] else 0
            dp_cur = [INF] * (m + 1)
            dp_cur[0] = 0
            for j in range(1, m + 1):
                # option1: skip this source char
                v = dp_prev[j]
                # option2: match this char to pattern[j-1] if equal
                if ch == pattern[j-1]:
                    # if we match, and this pos is in targetIndices, we incur cost of "keeping" it
                    v = min(v, dp_prev[j-1] + cost)
                dp_cur[j] = v
            dp_prev = dp_cur
        # dp_prev[m] = minimal number of targetIndices positions we had to keep
        min_kept = dp_prev[m]
        # we can remove the rest of targetIndices
        return len(targetIndices) - min_kept