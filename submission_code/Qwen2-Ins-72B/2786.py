class Solution:
    def longestSemiRepetitiveSubstring(self, s: str) -> int:
        max_len = 0
        for start in range(len(s)):
            for end in range(start + 1, len(s) + 1):
                substring = s[start:end]
                consecutive_repeats = sum(1 for i in range(1, len(substring)) if substring[i] == substring[i-1])
                if consecutive_repeats <= 1:
                    max_len = max(max_len, len(substring))
        return max_len