class Solution:
    def longestSemiRepetitiveSubstring(self, s: str) -> int:
        n = len(s)
        max_len = 0
        for i in range(n):
            dup_count = 0
            for j in range(i, n):
                if j > i and s[j] == s[j-1]:
                    dup_count += 1
                    if dup_count >= 2:
                        break
                length_here = j - i + 1
                if length_here > max_len:
                    max_len = length_here
        return max_len