class Solution:
    def longestSemiRepetitiveSubstring(self, s: str) -> int:
        n = len(s)
        max_len = 1
        start = 0
        consecutive_count = 0
        
        for end in range(n):
            if end > 0 and s[end] == s[end - 1]:
                consecutive_count += 1
            
            while consecutive_count > 1:
                if s[start] == s[start + 1]:
                    consecutive_count -= 1
                start += 1
            
            max_len = max(max_len, end - start + 1)
        
        return max_len