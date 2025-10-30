class Solution:
    def longestSemiRepetitiveSubstring(self, s: str) -> int:
        max_length = 1
        current_length = 1
        last_repeated_index = -1
        
        for i in range(1, len(s)):
            if s[i] == s[i - 1]:
                if last_repeated_index != -1:
                    current_length = i - last_repeated_index
                last_repeated_index = i
            current_length += 1
            max_length = max(max_length, current_length)
        
        return max_length