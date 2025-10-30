class Solution:
    def maximumLengthSubstring(self, s: str) -> int:
        max_length = 0
        left = 0
        count = [0] * 26
        for right in range(len(s)):
            char_index = ord(s[right]) - ord('a')
            count[char_index] += 1
            if count[char_index] > 2:
                while count[char_index] > 2:
                    left_char_index = ord(s[left]) - ord('a')
                    count[left_char_index] -= 1
                    left += 1
            max_length = max(max_length, right - left + 1)
        return max_length