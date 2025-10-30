class Solution:
    def maxRemovals(self, source: str, pattern: str, targetIndices: List[int]) -> int:
        def is_subsequence(s, p, removed):
            i, j = 0, 0
            while i < len(s) and j < len(p):
                if i in removed:
                    i += 1
                    continue
                if s[i] == p[j]:
                    j += 1
                i += 1
            return j == len(p)

        left, right = 0, len(targetIndices)
        
        while left <= right:
            mid = (left + right) // 2
            removed = set(targetIndices[:mid])
            
            if is_subsequence(source, pattern, removed):
                left = mid + 1
            else:
                right = mid - 1
        
        return right