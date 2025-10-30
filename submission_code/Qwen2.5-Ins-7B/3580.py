class Solution:
    def minStartingIndex(self, s: str, pattern: str) -> int:
        n, m = len(s), len(pattern)
        diff = 0
        for i in range(n - m + 1):
            j = 0
            while j < m and (s[i + j] == pattern[j] or diff == 1):
                if s[i + j] != pattern[j]:
                    diff += 1
                j += 1
            if j == m and diff <= 1:
                return i
            diff = 0
        return -1