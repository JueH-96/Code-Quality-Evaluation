class Solution:
    def longestSemiRepetitiveSubstring(self, s: str) -> int:
        def is_semi_repetitive(substring: str) -> bool:
            """Check if a string is semi-repetitive."""
            count = 0
            for i in range(len(substring) - 1):
                if substring[i] == substring[i + 1]:
                    count += 1
                    if count > 1:
                        return False
            return True

        max_length = 0
        for i in range(len(s)):
            for j in range(i + 1, len(s) + 1):
                substring = s[i:j]
                if is_semi_repetitive(substring):
                    max_length = max(max_length, len(substring))
        return max_length