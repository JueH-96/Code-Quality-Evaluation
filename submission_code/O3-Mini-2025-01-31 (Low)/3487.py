from typing import List

class Solution:
    def maxRemovals(self, source: str, pattern: str, targetIndices: List[int]) -> int:
        n = len(source)
        m = len(pattern)
        
        # Helper function to check if pattern is subsequence of source after removal.
        def canForm(k: int) -> bool:
            removed = [False] * n
            for i in range(k):
                removed[targetIndices[i]] = True
            
            # Check if pattern is subsequence in the remaining characters of source.
            j = 0  # pointer for pattern
            for i in range(n):
                if not removed[i] and j < m and source[i] == pattern[j]:
                    j += 1
                    if j == m:
                        break
            return j == m
        
        low, high, ans = 0, len(targetIndices), 0
        
        while low <= high:
            mid = (low + high) // 2
            if canForm(mid):
                ans = mid
                low = mid + 1
            else:
                high = mid - 1
                
        return ans