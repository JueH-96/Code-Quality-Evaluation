class Solution:
    def longestSemiRepetitiveSubstring(self, s: str) -> int:
        max_length = 0
        left = 0
        consecutive_pairs = 0
        
        for right in range(len(s)):
            if right > 0 and s[right] == s[right - 1]:
                consecutive_pairs += 1
            
            while consecutive_pairs > 1:
                if s[left] == s[left + 1]:
                    consecutive_pairs -= 1
                left += 1
            
            max_length = max(max_length, right - left + 1)
        
        return max_length