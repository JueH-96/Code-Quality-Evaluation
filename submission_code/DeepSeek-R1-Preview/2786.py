class Solution:
    def longestSemiRepetitiveSubstring(self, s: str) -> int:
        max_length = 1  # The minimum substring length is 1
        n = len(s)
        for i in range(n):
            current_duplicates = 0
            for j in range(i + 1, n):
                if s[j - 1] == s[j]:
                    current_duplicates += 1
                if current_duplicates > 1:
                    break
                current_length = j - i + 1
                if current_length > max_length:
                    max_length = current_length
        return max_length