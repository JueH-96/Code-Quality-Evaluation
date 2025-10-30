from typing import List

class Solution:
    def maxRemovals(self, source: str, pattern: str, targetIndices: List[int]) -> int:
        def is_subsequence(s: str, p: str) -> bool:
            it = iter(s)
            return all(char in it for char in p)

        left, right = 0, len(targetIndices)
        while left < right:
            mid = (left + right) // 2
            modified_source = ''.join(source[i] for i in range(len(source)) if i not in targetIndices[:mid + 1])
            if is_subsequence(modified_source, pattern):
                left = mid + 1
            else:
                right = mid
        
        return left