class Solution:
    def minStartingIndex(self, s: str, pattern: str) -> int:
        len_s = len(s)
        len_pattern = len(pattern)
        for i in range(len_s - len_pattern + 1):
            count = sum([s[i + j] != pattern[j] for j in range(len_pattern)])
            if count <= 1:
                return i
        return -1