class Solution:
    def minStartingIndex(self, s: str, pattern: str) -> int:
        n = len(s)
        m = len(pattern)
        for i in range(n - m + 1):
            if s[i:i+m] == pattern:
                return i
        return -1