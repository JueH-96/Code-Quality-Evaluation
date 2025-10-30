from collections import defaultdict

class Solution:
    def maximumLengthSubstring(self, s: str) -> int:
        freq = defaultdict(int)
        left = 0
        max_len = 0
        for right in range(len(s)):
            char = s[right]
            freq[char] += 1
            # Check if any character exceeds the count of 2
            while any(count > 2 for count in freq.values()):
                left_char = s[left]
                freq[left_char] -= 1
                if freq[left_char] == 0:
                    del freq[left_char]
                left += 1
            # Update the maximum length
            current_len = right - left + 1
            if current_len > max_len:
                max_len = current_len
        return max_len