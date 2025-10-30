from collections import defaultdict

class Solution:
    def maximumLengthSubstring(self, s: str) -> int:
        counts = defaultdict(int)
        left = 0
        max_len = 0
        for right in range(len(s)):
            current_char = s[right]
            counts[current_char] += 1
            
            # Shrink the window until the current character's count is at most 2
            while counts[current_char] > 2:
                left_char = s[left]
                counts[left_char] -= 1
                if counts[left_char] == 0:
                    del counts[left_char]
                left += 1
            
            # Update the maximum length found
            current_window_length = right - left + 1
            if current_window_length > max_len:
                max_len = current_window_length
        
        return max_len