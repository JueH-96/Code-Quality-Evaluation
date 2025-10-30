class Solution:
    def minStartingIndex(self, s: str, pattern: str) -> int:
        m = len(pattern)
        n = len(s)
        if m == 0:
            return 0
        for i in range(n - m + 1):
            # Check for exact match
            j = 0
            while j < m and s[i + j] == pattern[j]:
                j += 1
            if j == m:
                return i
            # Check for one mismatch
            # First mismatch is at j
            # Check if the rest of the characters match
            k = j + 1
            while k < m and s[i + k] == pattern[k]:
                k += 1
            if k == m:
                return i
        return -1