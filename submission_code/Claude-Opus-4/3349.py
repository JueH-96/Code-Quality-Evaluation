class Solution:
    def maximumLengthSubstring(self, s: str) -> int:
        char_count = {}
        left = 0
        max_length = 0
        
        for right in range(len(s)):
            # Add the current character to the window
            char = s[right]
            char_count[char] = char_count.get(char, 0) + 1
            
            # If any character appears more than twice, shrink the window
            while char_count[char] > 2:
                left_char = s[left]
                char_count[left_char] -= 1
                if char_count[left_char] == 0:
                    del char_count[left_char]
                left += 1
            
            # Update the maximum length
            max_length = max(max_length, right - left + 1)
        
        return max_length