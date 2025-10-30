class Solution:
    def longestSemiRepetitiveSubstring(self, s: str) -> int:
        n = len(s)
        max_length = 0
        for i in range(n):
            consecutive_pairs = 0
            for j in range(i, n):
                if j > i and s[j] == s[j - 1]:
                    consecutive_pairs += 1

                if consecutive_pairs <= 1:
                    max_length = max(max_length, j - i + 1)
        return max_length