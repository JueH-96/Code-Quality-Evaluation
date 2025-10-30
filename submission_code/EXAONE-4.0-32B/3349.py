class Solution:
    def maximumLengthSubstring(self, s: str) -> int:
        left = 0
        max_len = 0
        freq = {}
        n = len(s)
        
        for right in range(n):
            char = s[right]
            freq[char] = freq.get(char, 0) + 1
            
            while freq[char] > 2:
                left_char = s[left]
                freq[left_char] -= 1
                left += 1
                
            current_length = right - left + 1
            if current_length > max_len:
                max_len = current_length
                
        return max_len