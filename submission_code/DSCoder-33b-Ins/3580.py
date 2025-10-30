class Solution:
    def minStartingIndex(self, s: str, pattern: str) -> int:
        n, m = len(s), len(pattern)
        for i in range(n - m + 1):
            diff = sum(1 for j in range(m) if s[i + j] != pattern[j])
            if diff <= 1:
                return i
        return -1