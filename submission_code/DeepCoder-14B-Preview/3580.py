class Solution:
    def minStartingIndex(self, s: str, pattern: str) -> int:
        m = len(pattern)
        n = len(s)
        pattern_list = list(pattern)  # Convert pattern to list for faster access
        for i in range(n - m + 1):
            diff = 0
            for j in range(m):
                if s[i + j] != pattern_list[j]:
                    diff += 1
                    if diff > 1:
                        break
            if diff <= 1:
                return i
        return -1