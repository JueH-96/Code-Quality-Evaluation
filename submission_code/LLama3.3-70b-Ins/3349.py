class Solution:
    def maximumLengthSubstring(self, s: str) -> int:
        def is_valid(substring: str) -> bool:
            """Check if a substring has at most two occurrences of each character."""
            char_count = {}
            for char in substring:
                if char in char_count:
                    char_count[char] += 1
                else:
                    char_count[char] = 1
                if char_count[char] > 2:
                    return False
            return True

        max_length = 0
        for i in range(len(s)):
            for j in range(i + 1, len(s) + 1):
                substring = s[i:j]
                if is_valid(substring):
                    max_length = max(max_length, len(substring))
        return max_length