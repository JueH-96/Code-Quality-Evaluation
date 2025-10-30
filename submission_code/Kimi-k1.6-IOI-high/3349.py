from collections import defaultdict

class Solution:
    def maximumLengthSubstring(self, s: str) -> int:
        left = 0
        max_len = 0
        char_count = defaultdict(int)
        
        for right in range(len(s)):
            current_char = s[right]
            char_count[current_char] += 1
            
            # If the current character's count exceeds 2, move the left pointer
            while char_count[current_char] > 2:
                left_char = s[left]
                char_count[left_char] -= 1
                left += 1
            
            # Calculate the current window length and update max_len if necessary
            current_length = right - left + 1
            if current_length > max_len:
                max_len = current_length
        
        return max_len