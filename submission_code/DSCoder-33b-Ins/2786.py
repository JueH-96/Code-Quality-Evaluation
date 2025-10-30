class Solution:
    def longestSemiRepetitiveSubstring(self, s: str) -> int:
        max_len = 0
        curr_len = 1
        prev_char = s[0]
        for i in range(1, len(s)):
            if s[i] == prev_char:
                if curr_len == 1:
                    curr_len += 1
                else:
                    max_len = max(max_len, curr_len)
                    curr_len = 1
            else:
                curr_len += 1
            prev_char = s[i]
        max_len = max(max_len, curr_len)
        return max_len