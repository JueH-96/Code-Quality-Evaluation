class Solution:
    def maxRemovals(self, source: str, pattern: str, targetIndices: List[int]) -> int:
        def isSubsequence(s, p):
            i, j = 0, 0
            while i < len(s) and j < len(p):
                if s[i] == p[j]:
                    j += 1
                i += 1
            return j == len(p)

        def check(k):
            temp = list(source)
            for i in range(k):
                temp[targetIndices[i]] = ''
            return isSubsequence("".join(temp), pattern)

        left, right = 0, len(targetIndices)
        ans = 0
        while left <= right:
            mid = (left + right) // 2
            if check(mid):
                ans = mid
                left = mid + 1
            else:
                right = mid - 1
        return ans