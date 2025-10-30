class Solution:
    def maximumLengthSubstring(self, s: str) -> int:
        max_length = 0
        left = 0
        char_count = {}
        
        for right in range(len(s)):
            # Add current character to the window
            char_count[s[right]] = char_count.get(s[right], 0) + 1
            
            # Shrink window if any character appears more than 2 times
            while char_count[s[right]] > 2:
                char_count[s[left]] -= 1
                if char_count[s[left]] == 0:
                    del char_count[s[left]]
                left += 1
            
            # Update max length
            max_length = max(max_length, right - left + 1)
        
        return max_length