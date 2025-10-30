class Solution:
    def minStartingIndex(self, s: str, pattern: str) -> int:
        len_s = len(s)
        len_p = len(pattern)
        if len_p > len_s:
            return -1

        hamming = sum(1 for i in range(len_p) if s[i] != pattern[i])

        if hamming <= 1:
            return 0

        for i in range(1, len_s - len_p + 1):
            # Remove the mismatch from the previous window
            if s[i - 1] != pattern[0]:
                hamming -= 1
            # Add the mismatch for the new character in the window
            if s[i + len_p - 1] != pattern[-1]:
                hamming += 1

            if hamming <= 1:
                return i

        return -1