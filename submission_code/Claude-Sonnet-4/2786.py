class Solution:
    def longestSemiRepetitiveSubstring(self, s: str) -> int:
        n = len(s)
        if n <= 1:
            return n
        
        left = 0
        max_length = 1
        consecutive_pairs = 0
        
        for right in range(1, n):
            # Check if we have a new consecutive pair
            if s[right] == s[right - 1]:
                consecutive_pairs += 1
            
            # If we have more than 1 consecutive pair, shrink the window
            while consecutive_pairs > 1:
                if s[left] == s[left + 1]:
                    consecutive_pairs -= 1
                left += 1
            
            # Update max length
            max_length = max(max_length, right - left + 1)
        
        return max_length