class Solution:
    def maximumLengthSubstring(self, s: str) -> int:
        max_len = 0
        left = 0
        char_count = {}
        
        for right in range(len(s)):
            char = s[right]
            if char in char_count:
                char_count[char] += 1
            else:
                char_count[char] = 1
            
            while char_count[char] > 2:
                left_char = s[left]
                char_count[left_char] -= 1
                if char_count[left_char] == 0:
                    del char_count[left_char]
                left += 1
            
            current_len = right - left + 1
            if current_len > max_len:
                max_len = current_len
        
        return max_len