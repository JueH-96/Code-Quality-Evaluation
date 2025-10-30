from collections import defaultdict

class Solution:
    def maximumLengthSubstring(self, s: str) -> int:
        counts = defaultdict(int)
        left = 0
        max_len = 0
        for right in range(len(s)):
            counts[s[right]] += 1
            # Ensure the current character's count does not exceed 2
            while counts[s[right]] > 2:
                counts[s[left]] -= 1
                left += 1
            current_len = right - left + 1
            if current_len > max_len:
                max_len = current_len
        return max_len