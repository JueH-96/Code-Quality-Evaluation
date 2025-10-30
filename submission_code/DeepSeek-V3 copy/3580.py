class Solution:
    def minStartingIndex(self, s: str, pattern: str) -> int:
        n = len(s)
        m = len(pattern)
        if m > n:
            return -1
        for i in range(n - m + 1):
            mismatch = 0
            for j in range(m):
                if s[i + j] != pattern[j]:
                    mismatch += 1
                    if mismatch > 1:
                        break
            if mismatch <= 1:
                return i
        return -1