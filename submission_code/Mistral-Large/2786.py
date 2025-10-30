class Solution:
    def longestSemiRepetitiveSubstring(self, s: str) -> int:
        max_len = 1
        current_len = 1
        has_repetition = False

        for i in range(1, s.length):
            if s[i] == s[i - 1]:
                if has_repetition:
                    current_len = i - 1
                    has_repetition = False
                else:
                    has_repetition = True
            current_len += 1
            max_len = max(max_len, current_len)

        return max_len