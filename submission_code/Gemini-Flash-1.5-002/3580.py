class Solution:
    def minStartingIndex(self, s: str, pattern: str) -> int:
        len_s = len(s)
        len_p = len(pattern)
        if len_p > len_s:
            return -1

        min_index = -1
        for i in range(len_s - len_p + 1):
            substring = s[i:i + len_p]
            diff_count = 0
            for j in range(len_p):
                if substring[j] != pattern[j]:
                    diff_count += 1
            if diff_count <= 1:
                min_index = i
                break
        return min_index