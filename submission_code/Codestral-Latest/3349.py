class Solution:
    def maximumLengthSubstring(self, s: str) -> int:
        from collections import defaultdict

        max_length = 0
        left = 0
        char_count = defaultdict(int)

        for right in range(len(s)):
            char_count[s[right]] += 1

            while max(char_count.values()) > 2:
                char_count[s[left]] -= 1
                if char_count[s[left]] == 0:
                    del char_count[s[left]]
                left += 1

            max_length = max(max_length, right - left + 1)

        return max_length