class Solution:
    def longestSemiRepetitiveSubstring(self, s: str) -> int:
        n = len(s)
        max_len = 1  # Since s has at least length 1, the result is at least 1

        for i in range(n):
            repeated_pairs = 0
            for j in range(i, n):
                if j > i and s[j] == s[j - 1]:
                    repeated_pairs += 1
                if repeated_pairs > 1:
                    break
                max_len = max(max_len, j - i + 1)

        return max_len