from typing import List

class Solution:
    def maxRemovals(self, source: str, pattern: str, targetIndices: List[int]) -> int:
        def isSubsequence(s: str, p: str) -> bool:
            it = iter(s)
            return all(c in it for c in p)

        left, right = 0, len(targetIndices)
        while left < right:
            mid = (left + right + 1) // 2
            removed = [False] * len(source)
            for i in range(mid):
                removed[targetIndices[i]] = True
            new_source = ''.join(c for i, c in enumerate(source) if not removed[i])
            if isSubsequence(new_source, pattern):
                left = mid
            else:
                right = mid - 1
        return left