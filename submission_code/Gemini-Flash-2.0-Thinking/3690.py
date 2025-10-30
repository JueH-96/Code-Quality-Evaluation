import math

class Solution:
    def minLength(self, s: str, numOps: int) -> int:
        n = len(s)

        def check(maxLength):
            opsNeeded = 0
            i = 0
            while i < n:
                j = i
                while j < n and s[j] == s[i]:
                    j += 1
                length = j - i
                if length > maxLength:
                    opsNeeded += math.ceil(length / (maxLength + 1)) - 1
                i = j
            return opsNeeded <= numOps

        left, right = 1, n
        ans = n
        while left <= right:
            mid = (left + right) // 2
            if check(mid):
                ans = mid
                right = mid - 1
            else:
                left = mid + 1
        return ans