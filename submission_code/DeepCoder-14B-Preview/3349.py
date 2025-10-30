from collections import defaultdict

class Solution:
    def maximumLengthSubstring(self, s: str) -> int:
        count = defaultdict(int)
        max_len = 0
        left = 0
        for right in range(len(s)):
            char = s[right]
            count[char] += 1
            # Check if any character exceeds the maximum allowed count
            while True:
                over = False
                for c in 'abcdefghijklmnopqrstuvwxyz':
                    if count[c] > 2:
                        over = True
                        break
                if not over:
                    break
                # Move the left pointer to reduce the window size
                left_char = s[left]
                count[left_char] -= 1
                left += 1
            # Update the maximum length found
            current_len = right - left + 1
            if current_len > max_len:
                max_len = current_len
        return max_len