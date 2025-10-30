class Solution:
    def minStartingIndex(self, s: str, pattern: str) -> int:
        m, n = len(s), len(pattern)
        if m < n:
            return -1
        diff_count = 0
        for i in range(n):
            if s[i] != pattern[i]:
                diff_count += 1
        if diff_count <= 1:
            return 0
        for i in range(n, m):
            if s[i] != pattern[i]:
                diff_count += 1
            if s[i-n] != pattern[i-n]:
                diff_count -= 1
            if diff_count <= 1:
                return i - n + 1
        return -1