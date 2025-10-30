class Solution:
    def minStartingIndex(self, s: str, pattern: str) -> int:
        m, n = len(s), len(pattern)

        # Early return if pattern is longer than s
        if n > m:
            return -1

        # Function to check if s[i:i+n] is almost equal to pattern
        def is_almost_equal(i):
            mismatch_count = 0
            for j in range(n):
                if s[i + j] != pattern[j]:
                    mismatch_count += 1
                if mismatch_count > 1:
                    return False
            return True

        # Iterate through possible starting indices
        for i in range(m - n + 1):
            if is_almost_equal(i):
                return i

        return -1