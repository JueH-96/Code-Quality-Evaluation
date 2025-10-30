class Solution:
    def maxRemovals(self, source: str, pattern: str, targetIndices: List[int]) -> int:
        def is_subsequence(s, p):
            i = 0
            j = 0
            while i < len(s) and j < len(p):
                if s[i] == p[j]:
                    j += 1
                i += 1
            return j == len(p)

        left, right = 0, len(targetIndices)
        ans = 0
        while left <= right:
            mid = (left + right) // 2
            removed_indices = set(targetIndices[:mid])
            temp_source = ""
            for i in range(len(source)):
                if i not in removed_indices:
                    temp_source += source[i]
            if is_subsequence(temp_source, pattern):
                ans = mid
                left = mid + 1
            else:
                right = mid - 1
        return ans