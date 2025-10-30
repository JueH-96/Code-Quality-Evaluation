class Solution:
    def longestSemiRepetitiveSubstring(self, s: str) -> int:
        n = len(s)
        left = 0
        count = 0  # count of consecutive pairs in the current window
        max_length = 1
        
        for right in range(1, n):
            # Check if the new character forms a consecutive pair
            if s[right-1] == s[right]:
                count += 1
            
            # If count > 1, shrink the window from the left
            while count > 1:
                if s[left] == s[left + 1]:
                    count -= 1
                left += 1
            
            max_length = max(max_length, right - left + 1)
        
        return max_length