class Solution:
    def longestSemiRepetitiveSubstring(self, s: str) -> int:
        n = len(s)
        left = 0
        longest = 1  # Since the string length >= 1, at least single char is valid
        consecutive_count = 0  # Tracks how many consecutive pairs in current window
        
        for right in range(1, n):
            if s[right] == s[right - 1]:
                consecutive_count += 1
            
            # Shrink window from the left if we have more than one pair
            while consecutive_count > 1:
                if s[left] == s[left + 1]:
                    consecutive_count -= 1
                left += 1
            
            # Update the longest valid window size
            longest = max(longest, right - left + 1)
        
        return longest