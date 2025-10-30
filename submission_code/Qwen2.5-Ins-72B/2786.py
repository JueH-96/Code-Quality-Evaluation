class Solution:
    def longestSemiRepetitiveSubstring(self, s: str) -> int:
        max_length = 1
        start = 0
        repeat_count = 0
        
        for end in range(1, len(s)):
            if s[end] == s[end - 1]:
                repeat_count += 1
            
            while repeat_count > 1:
                if s[start] == s[start + 1]:
                    repeat_count -= 1
                start += 1
            
            max_length = max(max_length, end - start + 1)
        
        return max_length