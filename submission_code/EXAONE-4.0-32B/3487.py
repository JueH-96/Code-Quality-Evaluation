from typing import List

class Solution:
    def maxRemovals(self, source: str, pattern: str, targetIndices: List[int]) -> int:
        n = len(source)
        m = len(pattern)
        targetSet = set(targetIndices)
        NEG_INF = -10**9
        dp_prev = [NEG_INF] * (m + 1)
        dp_prev[0] = 0
        
        for i in range(1, n + 1):
            dp_curr = [NEG_INF] * (m + 1)
            for j in range(m + 1):
                best = dp_prev[j]
                idx = i - 1
                if idx in targetSet:
                    skip_val = dp_prev[j] + 1
                    if skip_val > best:
                        best = skip_val
                dp_curr[j] = best
                if j < m and source[idx] == pattern[j]:
                    if dp_prev[j] > dp_curr[j + 1]:
                        dp_curr[j + 1] = dp_prev[j]
            dp_prev = dp_curr
        
        return dp_prev[m]