from collections import defaultdict

class Solution:
    def maximumLengthSubstring(self, s: str) -> int:
        left = 0
        max_len = 0
        freq = defaultdict(int)
        for right in range(len(s)):
            char = s[right]
            freq[char] += 1
            # Ensure the current character's count does not exceed 2
            while freq[char] > 2:
                left_char = s[left]
                freq[left_char] -= 1
                left += 1
            # Update the maximum length found so far
            current_len = right - left + 1
            if current_len > max_len:
                max_len = current_len
        return max_len