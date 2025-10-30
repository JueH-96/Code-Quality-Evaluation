class Solution:
    def longestSemiRepetitiveSubstring(self, s: str) -> int:
        if len(s) <= 1:
            return len(s)
        
        max_length = 1
        left = 0
        consecutive_pairs = 0
        
        for right in range(1, len(s)):
            # Check if current character forms a consecutive pair with previous
            if s[right] == s[right - 1]:
                consecutive_pairs += 1
            
            # Shrink window from left while we have more than one consecutive pair
            while consecutive_pairs > 1:
                if s[left] == s[left + 1]:
                    consecutive_pairs -= 1
                left += 1
            
            # Update max length after ensuring we have at most one consecutive pair
            max_length = max(max_length, right - left + 1)
        
        return max_length