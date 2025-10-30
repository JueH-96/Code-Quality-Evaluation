class Solution:
    def longestSemiRepetitiveSubstring(self, s: str) -> int:
        max_len = 1  # Minimum possible length is 1 (single character)
        n = len(s)
        for L in range(n):
            current_pairs = 0
            for R in range(L + 1, n):
                if s[R] == s[R - 1]:
                    current_pairs += 1
                if current_pairs > 1:
                    break
                current_length = R - L + 1
                if current_length > max_len:
                    max_len = current_length
        return max_len