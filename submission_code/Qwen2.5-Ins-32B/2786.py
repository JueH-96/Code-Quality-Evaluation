class Solution:
    def longestSemiRepetitiveSubstring(self, s: str) -> int:
        max_len = 1
        current_len = 1
        repeat_count = 0
        
        for i in range(1, len(s)):
            if s[i] == s[i-1]:
                repeat_count += 1
                current_len += 1
                if repeat_count == 2:
                    max_len = max(max_len, current_len - 1)
                    current_len = 2
                    repeat_count = 1
            else:
                current_len += 1
                repeat_count = 1
            max_len = max(max_len, current_len)
        
        return max_len