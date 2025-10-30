class Solution:
    def longestSemiRepetitiveSubstring(self, s: str) -> int:
        max_length = 0
        for start in range(len(s)):
            for end in range(start + 1, len(s) + 1):
                substring = s[start:end]
                if substring.count(substring[0]) - substring.count(substring[0] * 2) <= 1:
                    max_length = max(max_length, len(substring))
        return max_length