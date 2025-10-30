class Solution:
    def maximumLengthSubstring(self, s: str) -> int:
        counts = {}
        left = 0
        max_length = 0

        for right in range(len(s)):
            char = s[right]
            counts[char] = counts.get(char, 0) + 1

            while counts[char] > 2:
                left_char = s[left]
                counts[left_char] -= 1
                left += 1

            current_window_length = right - left + 1
            if current_window_length > max_length:
                max_length = current_window_length

        return max_length