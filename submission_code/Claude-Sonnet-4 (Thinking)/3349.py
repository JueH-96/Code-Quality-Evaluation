class Solution:
    def maximumLengthSubstring(self, s: str) -> int:
        left = 0
        char_count = {}
        max_len = 0
        
        for right in range(len(s)):
            # Add current character to window
            char_count[s[right]] = char_count.get(s[right], 0) + 1
            
            # Shrink window if current character appears more than twice
            while char_count[s[right]] > 2:
                char_count[s[left]] -= 1
                if char_count[s[left]] == 0:
                    del char_count[s[left]]
                left += 1
            
            # Update maximum length
            max_len = max(max_len, right - left + 1)
        
        return max_len