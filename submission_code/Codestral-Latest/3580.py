class Solution:
    def minStartingIndex(self, s: str, pattern: str) -> int:
        m, n = len(s), len(pattern)

        for i in range(m - n + 1):
            diff_count = 0
            for j in range(n):
                if s[i + j] != pattern[j]:
                    diff_count += 1
                if diff_count > 1:
                    break
            if diff_count <= 1:
                return i

        return -1