class Solution:
    def maximumLengthSubstring(self, s: str) -> int:
        freq = [0] * 26
        left = 0
        max_len = 0
        for right in range(len(s)):
            char = s[right]
            freq[ord(char) - ord('a')] += 1
            # Ensure no character in the window exceeds two occurrences
            while freq[ord(char) - ord('a')] > 2:
                left_char = s[left]
                freq[ord(left_char) - ord('a')] -= 1
                left += 1
            # Update the maximum length found
            current_length = right - left + 1
            if current_length > max_len:
                max_len = current_length
        return max_len