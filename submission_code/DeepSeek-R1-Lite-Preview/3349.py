from collections import defaultdict

class Solution:
    def maximumLengthSubstring(self, s: str) -> int:
        freq = defaultdict(int)
        left = 0
        over_two = 0
        max_length = 0
        for right in range(len(s)):
            freq[s[right]] += 1
            if freq[s[right]] == 3:
                over_two += 1
            while over_two > 0:
                if freq[s[left]] == 3:
                    over_two -= 1
                freq[s[left]] -= 1
                if freq[s[left]] == 2:
                    over_two += 1  # This line is incorrect and should be removed
                left += 1
            max_length = max(max_length, right - left + 1)
        return max_length