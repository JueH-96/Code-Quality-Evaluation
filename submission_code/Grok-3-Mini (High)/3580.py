class Solution:
    def minStartingIndex(self, s: str, pattern: str) -> int:
        len_s = len(s)
        len_p = len(pattern)
        for i in range(len_s - len_p + 1):
            mismatch = 0
            for j in range(len_p):
                if s[i + j] != pattern[j]:
                    mismatch += 1
                    if mismatch > 1:
                        break  # No need to check further for this substring
            if mismatch <= 1:
                return i
        return -1