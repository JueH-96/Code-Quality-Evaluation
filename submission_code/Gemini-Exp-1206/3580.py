class Solution:
    def minStartingIndex(self, s: str, pattern: str) -> int:
        n = len(s)
        m = len(pattern)
        for i in range(n - m + 1):
            diff = 0
            for j in range(m):
                if s[i + j] != pattern[j]:
                    diff += 1
            if diff <= 1:
                return i
        return -1