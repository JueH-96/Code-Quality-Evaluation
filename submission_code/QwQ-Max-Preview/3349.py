from collections import defaultdict

class Solution:
    def maximumLengthSubstring(self, s: str) -> int:
        count = defaultdict(int)
        left = 0
        max_len = 0
        for right in range(len(s)):
            char = s[right]
            count[char] += 1
            # Ensure the current character's count does not exceed 2
            while count[char] > 2:
                left_char = s[left]
                count[left_char] -= 1
                left += 1
            current_len = right - left + 1
            if current_len > max_len:
                max_len = current_len
        return max_len