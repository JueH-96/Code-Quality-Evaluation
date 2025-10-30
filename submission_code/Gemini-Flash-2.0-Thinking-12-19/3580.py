class Solution:
    def minStartingIndex(self, s: str, pattern: str) -> int:
        n = len(s)
        m = len(pattern)
        if m > n:
            return -1

        for i in range(n - m + 1):
            substring = s[i:i+m]
            diff_count = 0
            for j in range(m):
                if substring[j] != pattern[j]:
                    diff_count += 1
            if diff_count <= 1:
                return i
        return -1