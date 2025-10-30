class Solution:
    def longestSemiRepetitiveSubstring(self, s: str) -> int:
        max_length = 0
        current_length = 1
        prev_char = None
        
        for i in range(1, len(s)):
            if s[i] == s[i-1]:
                current_length += 1
            else:
                max_length = max(max_length, current_length)
                current_length = 2 if prev_char == s[i] else 1
            prev_char = s[i]
        
        return max(max_length, current_length)