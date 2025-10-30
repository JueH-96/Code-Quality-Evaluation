from collections import Counter

class Solution:
    def maximumLengthSubstring(self, s: str) -> int:
        n = len(s)
        max_length = 0
        char_counts = Counter()
        left = 0

        for right in range(n):
            # Add the current character to the window
            char_counts[s[right]] += 1

            # Shrink the window from the left while the count of the character just added exceeds 2
            # If any character's count exceeds 2 after adding s[right], it must be s[right].
            while char_counts[s[right]] > 2:
                char_counts[s[left]] -= 1
                # No need to explicitly delete if count becomes 0, Counter handles it.
                left += 1

            # The current window s[left...right] is valid
            max_length = max(max_length, right - left + 1)

        return max_length