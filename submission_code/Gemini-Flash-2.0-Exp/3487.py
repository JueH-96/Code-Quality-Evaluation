class Solution:
    def maxRemovals(self, source: str, pattern: str, targetIndices: List[int]) -> int:
        def isSubsequence(s, p):
            i = 0
            j = 0
            while i < len(s) and j < len(p):
                if s[i] == p[j]:
                    j += 1
                i += 1
            return j == len(p)

        n = len(source)
        l, r = 0, len(targetIndices)
        ans = 0
        while l <= r:
            mid = (l + r) // 2
            temp_source = list(source)
            for i in range(mid):
                temp_source[targetIndices[i]] = ''
            temp_source = "".join(temp_source)
            if isSubsequence(temp_source, pattern):
                ans = mid
                l = mid + 1
            else:
                r = mid - 1
        return ans