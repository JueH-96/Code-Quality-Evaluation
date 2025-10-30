class Solution:
    def longestSemiRepetitiveSubstring(self, s: str) -> int:
        max_len = 0
        n = len(s)
        for i in range(n):
            for j in range(i, n):
                count = 0
                for k in range(i, j):
                    if s[k] == s[k + 1]:
                        count += 1
                if count <= 1:
                    current_len = j - i + 1
                    if current_len > max_len:
                        max_len = current_len
        return max_len