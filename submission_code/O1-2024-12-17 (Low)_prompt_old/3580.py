class Solution:
    def minStartingIndex(self, s: str, pattern: str) -> int:
        n, m = len(s), len(pattern)
        if m > n:
            return -1  # pattern longer than s, no valid index

        for start in range(n - m + 1):
            diff_count = 0
            for i in range(m):
                if s[start + i] != pattern[i]:
                    diff_count += 1
                    if diff_count > 1:
                        break
            if diff_count <= 1:
                return start
        return -1