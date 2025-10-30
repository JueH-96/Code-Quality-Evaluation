class Solution:
    def maximumLengthSubstring(self, s: str) -> int:
        freq = [0] * 26
        left = 0
        max_len = 0
        
        for right, char in enumerate(s):
            idx = ord(char) - ord('a')
            freq[idx] += 1
            while freq[idx] > 2:
                left_char = s[left]
                left_idx = ord(left_char) - ord('a')
                freq[left_idx] -= 1
                left += 1
            max_len = max(max_len, right - left + 1)
        
        return max_len