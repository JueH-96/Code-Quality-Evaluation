class Solution:
    def maximumLengthSubstring(self, s: str) -> int:
        counts = [0] * 26
        left = 0
        max_len = 0
        for right in range(len(s)):
            char = s[right]
            index = ord(char) - ord('a')
            counts[index] += 1
            if counts[index] > 2:
                while counts[index] > 2:
                    left_char = s[left]
                    counts[ord(left_char) - ord('a')] -= 1
                    left += 1
            current_len = right - left + 1
            if current_len > max_len:
                max_len = current_len
        return max_len