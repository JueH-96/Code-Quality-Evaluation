class Solution:
    def maximumLengthSubstring(self, s: str) -> int:
        from collections import defaultdict

        left = 0
        max_len = 0
        counts = defaultdict(int)

        for right in range(len(s)):
            counts[s[right]] += 1

            while counts[s[right]] > 2:
                counts[s[left]] -= 1
                left += 1

            current_window_length = right - left + 1
            if current_window_length > max_len:
                max_len = current_window_length

        return max_len