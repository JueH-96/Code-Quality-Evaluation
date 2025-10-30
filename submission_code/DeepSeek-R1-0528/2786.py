class Solution:
    def longestSemiRepetitiveSubstring(self, s: str) -> int:
        left = 0
        prev_rep = -1
        max_len = 0
        n = len(s)
        for right in range(n):
            if right > 0 and s[right] == s[right-1]:
                if prev_rep != -1:
                    left = prev_rep + 1
                prev_rep = right - 1
            max_len = max(max_len, right - left + 1)
        return max_len