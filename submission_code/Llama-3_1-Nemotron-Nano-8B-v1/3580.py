class Solution:
    def minStartingIndex(self, s: str, pattern: str) -> int:
        m = len(pattern)
        n = len(s)
        for i in range(n - m + 1):
            mismatches = 0
            for j in range(m):
                if s[i + j] != pattern[j]:
                    mismatches += 1
                    if mismatches > 1:
                        break
            if mismatches <= 1:
                return i
        return -1