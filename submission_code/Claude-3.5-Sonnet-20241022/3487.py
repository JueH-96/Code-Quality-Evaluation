class Solution:
    def maxRemovals(self, source: str, pattern: str, targetIndices: List[int]) -> int:
        def isSubsequence(s: str, p: str, removed: set) -> bool:
            i = j = 0
            while i < len(s) and j < len(p):
                if i in removed:
                    i += 1
                    continue
                if s[i] == p[j]:
                    j += 1
                i += 1
            return j == len(p)
        
        # Binary search on the number of removals
        left, right = 0, len(targetIndices)
        
        while left <= right:
            mid = (left + right) // 2
            # Try removing mid number of characters
            removed = set(targetIndices[:mid])
            
            # Check if pattern is still a subsequence after removals
            if isSubsequence(source, pattern, removed):
                left = mid + 1
            else:
                right = mid - 1
        
        return right