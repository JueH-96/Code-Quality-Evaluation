from typing import List

class Solution:
    def maxRemovals(self, source: str, pattern: str, targetIndices: List[int]) -> int:
        n = len(source)
        m = len(pattern)
        t = len(targetIndices)
        
        # Helper to check if we can remove the first k indices in targetIndices
        # and still have pattern as a subsequence of source.
        def can_remove(k: int) -> bool:
            removed = [False] * n
            for i in range(k):
                removed[targetIndices[i]] = True
            
            j = 0  # pointer in pattern
            for i in range(n):
                if removed[i]:
                    continue
                if j < m and source[i] == pattern[j]:
                    j += 1
                    if j == m:
                        return True
            return j == m
        
        # Binary search for the maximum k in [0..t] such that can_remove(k) is True.
        lo, hi = 0, t
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if can_remove(mid):
                lo = mid
            else:
                hi = mid - 1
        
        return lo