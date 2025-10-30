from typing import List

class Solution:
    def maxRemovals(self, source: str, pattern: str, targetIndices: List[int]) -> int:
        def can_remove(k: int) -> bool:
            removed = set(targetIndices[:k])
            p = 0
            for i, c in enumerate(source):
                if i in removed:
                    continue
                if p < len(pattern) and c == pattern[p]:
                    p += 1
                    if p == len(pattern):
                        break
            return p == len(pattern)
        
        left, right = 0, len(targetIndices)
        while left < right:
            mid = (left + right + 1) // 2
            if can_remove(mid):
                left = mid
            else:
                right = mid - 1
        return left